from executors.python import execute_function
import json

def calculate_reward(function_string: str, test_cases: list) -> float:
    """Calculate reward based on how many test cases the function passes."""
    if not test_cases:
        return 0.0
    
    passed_tests = 0
    
    for test_case in test_cases:
        try:
            actual_output = execute_function(function_string, test_case.inputs)
            
            # Check for execution errors
            if isinstance(actual_output, str) and "Error" in actual_output:
                passed_tests -= 1  # Penalty for execution error
                continue
            
            # Compare output
            try:
                actual_result = json.loads(actual_output.strip()) if isinstance(actual_output, str) else actual_output
            except:
                actual_result = str(actual_output).strip()
            
            if actual_result == test_case.output:
                passed_tests += 1
                
        except Exception:
            passed_tests -= 1  # Penalty for execution error
    
    return passed_tests / len(test_cases)
