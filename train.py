from unsloth import FastModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import torch
import json
import re
import os
from typing import List, Dict, Any
from executors.python import execute_function
from metrics.reward_functions import calculate_reward

# Initialize model and tokenizer
max_seq_length = 1024
max_prompt_length = 256

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it-qat",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

# Add LoRA adapters
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# System prompts
TEACHER_SYSTEM_PROMPT = """You are a problem proposer. Generate coding challenges with test cases.
The 'test_cases' field must contain 5 test cases with inputs and expected outputs.
Each test case should have 'inputs' (function arguments as dict) and 'output' (expected result).

You can use ```think ``` blocks to reason through problem design before generating your response.
Output your response in a ```solution ``` block with valid JSON format.

Example format:
```think
I need to create a problem about string manipulation. Let me think of a good challenge...
```

```solution
{
  "problem_statement": "Write a function that reverses each word in a sentence while keeping the word order.",
  "test_cases": [
    {"inputs": {"sentence": "hello world"}, "output": "olleh dlrow"},
    {"inputs": {"sentence": "python code"}, "output": "nohtyp edoc"},
    {"inputs": {"sentence": "a b c"}, "output": "a b c"},
    {"inputs": {"sentence": ""}, "output": ""},
    {"inputs": {"sentence": "single"}, "output": "elgnis"}
  ]
}
```"""

STUDENT_SYSTEM_PROMPT = """You are a coder. Given a coding challenge, write Python code to solve it.
Your code must be optimized and clean. If you determine that the problem is impossible to solve 
or the test cases are inconsistent, set "code" to null and explain why in the explanation field.

You can use ```think ``` blocks to reason through the problem before providing your solution.
Output your response in a ```solution ``` block with valid JSON format containing "code" and "explanation" fields.

Example format:
```think
I need to reverse each word while keeping word order. I can split by spaces, reverse each word, then join back.
```

```solution
{
  "code": "def reverse_words(sentence):\n    return ' '.join(word[::-1] for word in sentence.split(' '))",
  "explanation": "Split the sentence by spaces, reverse each word using slicing [::-1], then join back with spaces."
}
```"""

def generate_with_model(messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
    """Generate response using the model"""
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def create_teacher_dataset() -> Dict[str, Any]:
    """Generate a coding problem using the teacher"""
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "Generate a unique coding challenge with 5 test cases."}
    ]
    
    return {
        "prompt": messages
    }

def create_student_dataset(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create student training data from a problem"""
    try:
        # Extract problem from generated text (would need parsing logic)
        # For now, create a sample problem format
        test_cases_str = "Sample test cases here"  # This would be parsed from teacher output
        
        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: Solve this coding challenge\n\nTest Cases:\n{test_cases_str}\n\nPlease provide your solution."}
        ]
        
        return {
            "prompt": messages,
            "problem_data": problem_data
        }
    except Exception as e:
        print(f"Error creating student dataset: {e}")
        return None

# Think block detection and solution block extraction
def detect_think_blocks(text: str) -> tuple[bool, int]:
    """Detect ```think ``` blocks and return (has_think_blocks, num_blocks)"""
    import re
    think_pattern = r'```think\s*(.*?)```'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)
    num_blocks = len(matches)
    
    return num_blocks > 0, num_blocks

def extract_solution_json(text: str) -> dict:
    """Extract JSON from ```solution ``` blocks"""
    import re
    solution_pattern = r'```solution\s*(.*?)```'
    matches = re.findall(solution_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        try:
            return json.loads(matches[0].strip())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to parse the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}

def solution_block_format_reward(prompts, completions, **kwargs):
    """Reward for having exactly one solution block at the very end of response"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        # Find all solution blocks
        solution_pattern = r'```solution\s*(.*?)```'
        matches = re.findall(solution_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if len(matches) == 1:
            # Exactly one solution block found
            score += 2.0
            
            # Check if it's at the very end
            # Find the last occurrence of ```solution
            last_solution_start = response.lower().rfind('```solution')
            if last_solution_start != -1:
                # Find the closing ``` after the last ```solution
                closing_pos = response.find('```', last_solution_start + 11)  # 11 = len('```solution')
                if closing_pos != -1:
                    # Check if there's only whitespace after the closing ```
                    after_solution = response[closing_pos + 3:].strip()
                    if not after_solution:
                        score += 1.0  # Bonus for being at the very end
                    else:
                        score -= 0.5  # Small penalty for content after solution block
        elif len(matches) == 0:
            score = -2.0  # Heavy penalty for no solution block
        else:
            score = -1.0  # Penalty for multiple solution blocks
            
        scores.append(score)
    
    return scores

def think_block_reward_function(prompts, completions, **kwargs):
    """Reward function specifically for ```think ``` block usage"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        has_think, num_blocks = detect_think_blocks(response)
        
        if has_think:
            # Base reward for using think blocks
            score += 1.0
            
            # Check if think blocks have meaningful content
            think_pattern = r'```think\s*(.*?)```'
            matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)
            non_empty_blocks = [m for m in matches if m.strip()]
            
            if non_empty_blocks:
                # Bonus for having content in think blocks
                score += 0.5 * len(non_empty_blocks)
            
        # No penalty for no think blocks - it's completely optional
            
        scores.append(score)
    
    return scores

# Student reward functions - split into multiple metrics

def student_json_format_reward(prompts, completions, **kwargs):
    """Reward for valid JSON format in student responses"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        response_data = extract_solution_json(response)
        
        if response_data:  # Successfully extracted JSON
            score += 1.0  # Valid JSON format
            
            # Check for required fields
            if "code" in response_data:
                score += 0.5
            if "explanation" in response_data:
                score += 0.5
        else:
            score = -2.0  # Heavy penalty for invalid JSON
            
        scores.append(score)
    
    return scores

def student_code_correctness_reward(prompts, completions, problem_data=None, **kwargs):
    """Reward based on code correctness using test cases"""
    scores = []
    
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        score = 0.0
        response = completion[0]["content"]
        
        response_data = extract_solution_json(response)
        
        if response_data:
            code = response_data.get("code")
            
            if code is None:
                score = -1.0  # Penalty for giving up
            else:
                # Get test cases for this problem
                if problem_data and i < len(problem_data):
                    test_cases = problem_data[i].get("test_cases", [])
                    if test_cases:
                        # Use our calculate_reward function
                        score = calculate_reward(response, test_cases)
                        # Scale the score (calculate_reward returns 0-1 range)
                        score = score * 4.0  # Scale to 0-4 range for higher weight
                    else:
                        score = 2.0  # Partial credit for valid code without test cases
                else:
                    score = 2.0  # Partial credit when no test cases available
        else:
            score = 0.0  # No reward for invalid JSON (handled by format reward)
            
        scores.append(score)
    
    return scores

def student_explanation_reward(prompts, completions, **kwargs):
    """Reward for providing explanations"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        response_data = extract_solution_json(response)
        
        if response_data:
            explanation = response_data.get("explanation", "")
            
            if explanation and explanation.strip():
                score += 1.0  # Has explanation
                
                # Bonus for longer explanations (indicates more thought)
                if len(explanation.strip()) > 50:
                    score += 0.5
        else:
            score = 0.0  # No reward for invalid JSON
            
        scores.append(score)
    
    return scores

def student_think_block_reward(prompts, completions, **kwargs):
    """Reward for using think blocks in reasoning"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        has_think, num_blocks = detect_think_blocks(response)
        
        if has_think:
            score += 1.0  # Base reward for using think blocks
            
            # Check if think blocks have meaningful content
            think_pattern = r'```think\s*(.*?)```'
            matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)
            non_empty_blocks = [m for m in matches if m.strip()]
            
            if non_empty_blocks:
                # Bonus for having content in think blocks
                score += 0.5 * len(non_empty_blocks)
            
        scores.append(score)
    
    return scores

# Teacher reward functions - split into multiple metrics

def teacher_json_format_reward(prompts, completions, **kwargs):
    """Reward for valid JSON format in teacher responses"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        problem_data = extract_solution_json(response)
        
        if problem_data:  # Successfully extracted JSON
            score += 1.0  # Valid JSON format
            
            # Check for required fields
            if "problem_statement" in problem_data:
                score += 0.5
            if "test_cases" in problem_data:
                score += 0.5
        else:
            score = -2.0  # Heavy penalty for invalid JSON
            
        scores.append(score)
    
    return scores

def teacher_problem_format_reward(prompts, completions, **kwargs):
    """Reward for proper problem structure and test cases"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        problem_data = extract_solution_json(response)
        
        if problem_data:
            test_cases = problem_data.get("test_cases", [])
            problem_statement = problem_data.get("problem_statement", "")
            
            # Reward for having exactly 5 test cases
            if len(test_cases) == 5:
                score += 2.0
            elif len(test_cases) >= 3:
                score += 1.0  # Partial credit for having some test cases
            else:
                score = -1.0  # Penalty for too few test cases
                
            # Reward for having a problem statement
            if problem_statement and len(problem_statement.strip()) > 20:
                score += 1.0
            elif problem_statement:
                score += 0.5
                
            # Check test case quality
            valid_test_cases = 0
            for tc in test_cases:
                if isinstance(tc, dict) and "inputs" in tc and "output" in tc:
                    valid_test_cases += 1
                    
            if valid_test_cases == len(test_cases) and len(test_cases) > 0:
                score += 1.0  # All test cases are properly formatted
        else:
            score = 0.0  # No reward for invalid JSON (handled by format reward)
            
        scores.append(score)
    
    return scores

def teacher_student_solvability_reward(prompts, completions, **kwargs):
    """Reward based on how well student can solve the generated problem"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        problem_data = extract_solution_json(response)
        
        if problem_data:
            test_cases = problem_data.get("test_cases", [])
            problem_statement = problem_data.get("problem_statement", "")
            
            if not test_cases or not problem_statement:
                score = -1.0
                scores.append(score)
                continue
            
            # Generate student solution for this problem
            test_cases_str = "\n".join([
                f"Input: {tc.get('inputs', {})}, Expected Output: {tc.get('output', '')}"
                for tc in test_cases[:3]  # Show only first 3 to student
            ])
            
            student_messages = [
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Problem: {problem_statement}\n\nTest Cases:\n{test_cases_str}\n\nPlease provide your solution."}
            ]
            
            student_response = generate_with_model(student_messages)
            
            student_data = extract_solution_json(student_response)
            
            if student_data:
                if student_data.get("code") is None:
                    # Heavy penalty if student can't solve teacher's problem
                    score = -2.0
                else:
                    # Use calculate_reward to evaluate student's solution
                    from models.models import TestCase
                    test_case_objects = []
                    for tc in test_cases:
                        test_case_objects.append(TestCase(
                            inputs=tc.get("inputs", {}),
                            output=tc.get("output", "")
                        ))
                    
                    student_score = calculate_reward(student_response, test_case_objects)
                    
                    # Teacher gets rewarded based on how well student performs
                    if student_score >= 0.8:  # Student solved it well
                        score = 3.0
                    elif student_score >= 0.6:  # Student partially solved it
                        score = 2.0
                    elif student_score >= 0.4:  # Student struggled but made progress
                        score = 1.0
                    else:  # Student failed to solve it
                        score = -0.5
            else:
                score = -1.0
        else:
            print(f"Error in teacher solvability reward: No valid solution block found")
            score = -1.0
            
        scores.append(score)
    
    return scores

def teacher_think_block_reward(prompts, completions, **kwargs):
    """Reward for using think blocks in teacher responses"""
    scores = []
    
    for prompt, completion in zip(prompts, completions):
        score = 0.0
        response = completion[0]["content"]
        
        has_think, num_blocks = detect_think_blocks(response)
        
        if has_think:
            score += 0.8  # Base reward for using think blocks
            
            # Check if think blocks have meaningful content
            think_pattern = r'```think\s*(.*?)```'
            matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)
            non_empty_blocks = [m for m in matches if m.strip()]
            
            if non_empty_blocks:
                # Bonus for having content in think blocks
                score += 0.3 * len(non_empty_blocks)
            
        scores.append(score)
    
    return scores

# Training configuration
def get_training_args(output_dir, save_steps=10):
    return GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=100,  # 100 iterations per training phase
        save_steps=save_steps,  # Save checkpoint every N steps
        max_grad_norm=0.1,
        report_to="none",
        output_dir=output_dir,
        save_strategy="steps",
    )

def print_generation_examples():
    """Print 2 generation examples before training"""
    print("\n=== Generation Examples Before Training ===")
    
    # Example 1: Teacher generation
    print("\n--- Example 1: Teacher Problem Generation ---")
    teacher_messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "Generate a unique coding challenge with 5 test cases."}
    ]
    teacher_example = generate_with_model(teacher_messages, max_tokens=512)
    print("Teacher Output:")
    print(teacher_example)
    
    # Calculate and display teacher rewards
    print("\nTeacher Rewards:")
    teacher_completions = [[{"content": teacher_example}]]
    teacher_prompts = [teacher_messages]
    
    solution_reward = solution_block_format_reward(teacher_prompts, teacher_completions)[0]
    json_reward = teacher_json_format_reward(teacher_prompts, teacher_completions)[0]
    format_reward = teacher_problem_format_reward(teacher_prompts, teacher_completions)[0]
    think_reward = teacher_think_block_reward(teacher_prompts, teacher_completions)[0]
    
    print(f"  Solution Block Reward: {solution_reward:.2f}")
    print(f"  JSON Format Reward: {json_reward:.2f}")
    print(f"  Problem Format Reward: {format_reward:.2f}")
    print(f"  Think Block Reward: {think_reward:.2f}")
    print(f"  Total Teacher Reward: {solution_reward + json_reward + format_reward + think_reward:.2f}")
    
    # Example 2: Student generation
    print("\n--- Example 2: Student Code Solution ---")
    student_messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": """Problem: Write a function that takes a list of integers and returns the sum of all even numbers.

Test Cases:
Input: {'nums': [1, 2, 3, 4, 5, 6]}, Expected Output: 12
Input: {'nums': [1, 3, 5]}, Expected Output: 0
Input: {'nums': [2, 4, 6, 8]}, Expected Output: 20

Please provide your solution."""}
    ]
    student_example = generate_with_model(student_messages, max_tokens=512)
    print("Student Output:")
    print(student_example)
    
    # Calculate and display student rewards
    print("\nStudent Rewards:")
    student_completions = [[{"content": student_example}]]
    student_prompts = [student_messages]
    
    solution_reward = solution_block_format_reward(student_prompts, student_completions)[0]
    json_reward = student_json_format_reward(student_prompts, student_completions)[0]
    explanation_reward = student_explanation_reward(student_prompts, student_completions)[0]
    think_reward = student_think_block_reward(student_prompts, student_completions)[0]
    
    print(f"  Solution Block Reward: {solution_reward:.2f}")
    print(f"  JSON Format Reward: {json_reward:.2f}")
    print(f"  Explanation Reward: {explanation_reward:.2f}")
    print(f"  Think Block Reward: {think_reward:.2f}")
    print(f"  Total Student Reward (w/o correctness): {solution_reward + json_reward + explanation_reward + think_reward:.2f}")
    
    print("\n" + "="*50 + "\n")

def train_continuous():
    """Main training loop with continuous dataset generation"""
    cycle = 0
    
    # Create checkpoint directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Print generation examples before training
    print_generation_examples()
    
    while True:
        print(f"\n=== Training Cycle {cycle + 1} ===")
        
        # Phase 1: Train Student for 100 iterations
        print("Phase 1: Training Student...")
        
        # Generate dynamic dataset for student training
        student_data = []
        for i in range(100):  # Generate 100 problems for student training
            teacher_data = create_teacher_dataset()
            student_sample = create_student_dataset(teacher_data)
            if student_sample:
                student_data.append(student_sample)
        
        if student_data:
            student_dataset = Dataset.from_list(student_data)
            
            # Student training with frequent checkpoints
            student_args = get_training_args(
                output_dir=f"outputs/student_cycle_{cycle}",
                save_steps=10  # Save every 10 steps (epoch-like checkpoints)
            )
            
            student_trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[
                    solution_block_format_reward,
                    student_json_format_reward,
                    student_code_correctness_reward,
                    student_explanation_reward,
                    student_think_block_reward,
                ],
                args=student_args,
                train_dataset=student_dataset,
            )
            
            print("Starting student training...")
            student_trainer.train()
            print("Student training completed.")
            
            # Save student phase checkpoint
            model.save_pretrained(f"checkpoints/student_cycle_{cycle}")
            tokenizer.save_pretrained(f"checkpoints/student_cycle_{cycle}")
        
        # Phase 2: Train Teacher for 100 iterations  
        print("Phase 2: Training Teacher...")
        
        # Generate dynamic dataset for teacher training
        teacher_data = []
        for i in range(100):  # Generate 100 teacher samples
            teacher_sample = create_teacher_dataset()
            teacher_data.append(teacher_sample)
        
        teacher_dataset = Dataset.from_list(teacher_data)
        
        # Teacher training with frequent checkpoints
        teacher_args = get_training_args(
            output_dir=f"outputs/teacher_cycle_{cycle}",
            save_steps=10  # Save every 10 steps (epoch-like checkpoints)
        )
        
        teacher_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                solution_block_format_reward,
                teacher_json_format_reward,
                teacher_problem_format_reward,
                teacher_student_solvability_reward,
                teacher_think_block_reward,
            ],
            args=teacher_args,
            train_dataset=teacher_dataset,
        )
        
        print("Starting teacher training...")
        teacher_trainer.train()
        print("Teacher training completed.")
        
        # Save teacher phase checkpoint
        model.save_pretrained(f"checkpoints/teacher_cycle_{cycle}")
        tokenizer.save_pretrained(f"checkpoints/teacher_cycle_{cycle}")
        
        # Save checkpoint
        model.save_pretrained(f"checkpoints/cycle_{cycle}")
        tokenizer.save_pretrained(f"checkpoints/cycle_{cycle}")
        
        cycle += 1
        
        # Optional: Add stopping condition
        if cycle >= 10:  # Stop after 10 cycles
            break

if __name__ == "__main__":
    print("Starting continuous training...")
    train_continuous()
    print("Training completed!")