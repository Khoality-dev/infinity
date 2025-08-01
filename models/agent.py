from abc import ABC, abstractmethod
from ollama import Client
from pydantic import BaseModel
from typing import List, Optional

client = Client(
    host='http://10.0.0.122:11434',
)

class Agent(ABC):
    def __init__(self, model: str = 'gemma3:12b-it-qat'):
        self.model = model
        self.client = client
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
    
    def _chat(self, messages: List[dict], format_schema: Optional[BaseModel] = None):
        chat_params = {
            'model': self.model,
            'messages': messages,
            'stream': False
        }
        
        if format_schema:
            chat_params['format'] = format_schema.model_json_schema()
        
        response = self.client.chat(**chat_params)
        
        return response.message.content