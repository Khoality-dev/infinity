from models.agent import Agent
from models.models import CodingChallenge, Solution
import json

class Student(Agent):
    def __init__(self, model: str = 'gemma3:12b-it-qat'):
        super().__init__(model)
        self.system_prompt = """
        You are a coder. Given a coding challenge, write Python code to solve it. Your code must be optimized and clean. If you determine that the problem is impossible to solve or the test cases are inconsistent, set "code" to null and explain why in the explanation field.
        
        You can use ```think ``` blocks to reason through the problem before providing your solution.
        
        """
    
    def __call__(self, challenge: CodingChallenge) -> Solution:
        test_cases_str = "\n".join([
            f"Input: {tc.inputs}, Expected Output: {tc.output}"
            for tc in challenge.test_cases[:3]
        ])
        
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': f"""Problem: {challenge.problem_statement}

Test Cases:
{test_cases_str}

Please provide your solution."""
            }
        ]
        
        content = self._chat(messages, Solution)
        
        try:
            solution_data = json.loads(content)
            return Solution(**solution_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse solution: {e}")