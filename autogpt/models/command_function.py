import json
from typing import Any


class CommandFunction:
    """Represents a "function" in OpenAI, which is mapped to a Command in Auto-GPT"""

    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @property
    def __dict__(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
