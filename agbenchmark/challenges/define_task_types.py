import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, validator


class DifficultyLevel(Enum):
    interface = "interface"
    basic = "basic"
    novice = "novice"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    human = "human"


# map from enum to difficulty level (numeric)
DIFFICULTY_MAP = {
    DifficultyLevel.interface: 1,
    DifficultyLevel.basic: 2,
    DifficultyLevel.novice: 3,
    DifficultyLevel.intermediate: 4,
    DifficultyLevel.advanced: 5,
    DifficultyLevel.expert: 6,
    DifficultyLevel.human: 7,
}


class Info(BaseModel):
    difficulty: DifficultyLevel
    description: str
    side_effects: List[str]

    @validator("difficulty", pre=True)
    def difficulty_to_enum(cls: "Info", v: str | DifficultyLevel) -> DifficultyLevel:
        """Convert a string to an instance of DifficultyLevel."""
        if isinstance(v, DifficultyLevel):
            return v

        if isinstance(v, str):
            try:
                return DifficultyLevel(v.lower())
            except ValueError:
                pass

        raise ValueError(f"Cannot convert {v} to DifficultyLevel.")


class Ground(BaseModel):
    answer: str
    should_contain: Optional[List[str]] = None
    should_not_contain: Optional[List[str]] = None
    files: List[str]
    type: str


class ChallengeData(BaseModel):
    name: str
    category: List[str]
    task: str
    dependencies: List[str]
    ground: Ground
    info: Info

    def serialize(self, path: str) -> None:
        with open(path, "w") as file:
            file.write(self.json())

    @staticmethod
    def deserialize(path: str) -> "ChallengeData":
        # this script is in root/agbenchmark/challenges/define_task_types.py
        script_dir = Path(__file__).resolve().parent.parent.parent
        path = str(script_dir / path)

        print("Deserializing", path)

        with open(path, "r") as file:
            data = json.load(file)
        return ChallengeData(**data)
