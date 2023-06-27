from pydantic import BaseModel
from typing import List, Optional
import json
import os


class Mock(BaseModel):
    mock_func: str
    mock_task: Optional[str] = None


class Info(BaseModel):
    difficulty: str
    description: str
    side_effects: List[str]


class Ground(BaseModel):
    answer: str
    should_contain: Optional[List[str]] = None
    should_not_contain: Optional[List[str]] = None
    files: List[str]


class ChallengeData(BaseModel):
    name: str
    category: List[str]
    task: str
    dependencies: List[str]
    ground: Ground
    mock: Optional[Mock] = None
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
