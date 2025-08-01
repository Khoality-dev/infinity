from pydantic import BaseModel
from typing import List, Union, Optional

class TestCase(BaseModel):
    inputs: dict[str, Union[str, int, float, bool, List[Union[str, int, float]], dict]]
    output: Union[str, int, float, bool, List[Union[str, int, float]], dict]

class CodingChallenge(BaseModel):
    problem_statement: str
    test_cases: List[TestCase]

class Solution(BaseModel):
    code: Optional[str]
    explanation: str