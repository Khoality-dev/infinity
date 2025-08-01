from models.agent import Agent
from models.models import CodingChallenge
import json

class Teacher(Agent):
    def __init__(self, model: str = 'gemma3:12b-it-qat'):
        super().__init__(model)
        self.system_prompt = """You are a problem proposer, your task is to provide hard coding challenges. 
The 'test_cases' field must contain 5 test cases with inputs and expected outputs.
Each test case should have 'inputs' (function arguments) and 'output' (expected result)."""
    
    def __call__(self) -> CodingChallenge:
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': 'Generate a unique coding challenge'
            }
        ]
        
        content = self._chat(messages, CodingChallenge)
        
        try:
            challenge_data = json.loads(content)
            return CodingChallenge(**challenge_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse challenge: {e}")