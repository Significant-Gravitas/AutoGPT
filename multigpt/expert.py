import yaml
import os
from typing import List
from autogpt.config.ai_config import AIConfig


class Expert(AIConfig):
    def to_yaml(self) -> str:
        return yaml.dump(self.__dict__)

    def to_string(self) -> str:
        return f"Name: {self.name}, Role: {self.role}, Goals: {self.goals}"
