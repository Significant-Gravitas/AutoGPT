import yaml
from typing import List

class Expert:
    name: str
    role: str
    goals: List[str]

    def __init__(self, name: str, role: str, goals: List[str]) -> None:
        self.name = name
        self.role = role
        self.goals = goals
    
    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)
    
    def to_string(self) -> str:
        return f"Name: {self.name}, Role: {self.role}, Goals: {self.goals}"
    