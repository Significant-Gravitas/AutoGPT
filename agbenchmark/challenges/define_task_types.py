from pydantic import BaseModel
from typing import List, Optional
import json
import os


class Info(BaseModel):
    difficulty: str
    description: str
    side_effects: List[str]


class Ground(BaseModel):
    answer: str
    should_contain: Optional[List[str]]
    should_not_contain: Optional[List[str]]
    files: List[str]


class ChallengeData(BaseModel):
    category: List[str]
    task: str
    ground: Ground
    mock_func: Optional[str] = None
    info: Info

    def serialize(self, path: str) -> None:
        with open(path, "w") as file:
            file.write(self.json())

    @staticmethod
    def deserialize(path: str) -> "ChallengeData":
        print("Deserializing", path)
        with open(path, "r") as file:
            data = json.load(file)
        return ChallengeData(**data)
