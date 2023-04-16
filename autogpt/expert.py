import yaml
import os
from typing import List
from autogpt.config.ai_config import AIConfig

class Expert(AIConfig):
    # name: str
    # role: str
    # goals: List[str]

    # def __init__(self, name: str, role: str, goals: List[str]) -> None:
    #     self.name = name
    #     self.role = role
    #     self.goals = goals

    SAVE_FILE = os.path.join(os.path.dirname(__file__), "..", "expert_settings.yaml")
    
    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)
    
    def to_string(self) -> str:
        return f"Name: {self.name}, Role: {self.role}, Goals: {self.goals}"
    
    