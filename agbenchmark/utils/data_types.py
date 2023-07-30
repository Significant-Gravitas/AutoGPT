import glob
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator, validator


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

STRING_DIFFICULTY_MAP = {e.value: DIFFICULTY_MAP[e] for e in DifficultyLevel}


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


class Eval(BaseModel):
    type: str
    scoring: Optional[str]
    template: Optional[str]
    examples: Optional[str]

    @validator("scoring", "template", always=True)
    def validate_eval_fields(cls, v, values, field):
        if "type" in values and values["type"] == "llm":
            if v is None:
                raise ValueError(f"{field.name} must be provided when type is 'llm'")
        else:
            if v is not None:
                raise ValueError(f"{field.name} should only exist when type is 'llm'")
        return v

    @validator("scoring")
    def validate_scoring(cls, v):
        if v is not None and v not in ["percentage", "scale", "binary"]:
            raise ValueError(
                "scoring must be either 'percentage', 'scale', or 'binary'"
            )
        return v

    @validator("template")
    def validate_template(cls, v):
        if v is not None and v not in ["rubric", "reference", "question", "custom"]:
            raise ValueError(
                "template must be either 'rubric', 'reference', 'question', or 'custom'"
            )
        return v


class Ground(BaseModel):
    answer: str
    should_contain: Optional[List[str]] = None
    should_not_contain: Optional[List[str]] = None
    files: List[str]
    eval: Eval


class ChallengeData(BaseModel):
    name: str
    category: List[str]
    task: str
    dependencies: List[str]
    cutoff: int
    ground: Ground | Dict[str, Ground]
    info: Info | Dict[str, Info]

    def serialize(self, path: str) -> None:
        with open(path, "w") as file:
            file.write(self.json())

    def get_data(self) -> dict:
        return self.dict()

    @staticmethod
    def get_json_from_path(json_path: Path | str) -> dict:
        path = Path(json_path).resolve()
        with open(path, "r") as file:
            data = json.load(file)
        return data

    @staticmethod
    def deserialize(path: str) -> "ChallengeData":
        # this script is in root/agbenchmark/utils/define_task_types.py
        script_dir = Path(__file__).resolve().parent.parent.parent
        json_path = script_dir / Path(path)

        with open(json_path, "r") as file:
            data = json.load(file)

        return ChallengeData(**data)


class SuiteConfig(BaseModel):
    same_task: bool
    reverse_order: Optional[bool] = None
    prefix: str
    task: Optional[str] = None
    cutoff: Optional[int] = None
    dependencies: Optional[List[str]] = None
    shared_category: Optional[List[str]] = None
    info: Optional[Dict[str, Info]] = None
    ground: Optional[Dict[str, Ground]] = None

    @root_validator
    def check_attributes(cls: Any, values: Dict[str, Any]) -> Dict[str, Any]:
        same_task = values.get("same_task")
        if same_task:
            if (
                values.get("task") is None
                or values.get("cutoff") is None
                or values.get("dependencies") is None
                or values.get("shared_category") is None
            ):
                raise ValueError(
                    f"task, cutoff, dependencies, and shared_category must be provided when same_task is True for test {cls.prefix}."
                )
        else:
            if values.get("reverse_order") is None:
                raise ValueError(
                    f"reverse_order must be provided when same_task is False for test {cls.prefix}."
                )

        return values

    @staticmethod
    def suite_data_if_suite(json_path: Path) -> Optional["SuiteConfig"]:
        """Return the suite data if the path is in a suite."""
        if SuiteConfig.check_if_suite(json_path):
            return SuiteConfig.deserialize_from_test_data(json_path)
        else:
            return None

    @staticmethod
    def check_if_suite(json_path: Path) -> bool:
        """Check if the json file is in a suite."""

        # if its in a suite, suite.json is in the parent suite/suite.json & 1_challenge/data.json
        suite_path = json_path.parent.parent / "suite.json"

        # validation and loading data from suite.json
        return suite_path.exists()

    @staticmethod
    def deserialize_from_test_data(data_path: Path) -> "SuiteConfig":
        """Deserialize from a children path when children and order of children does not matter."""

        suite_path = data_path.parent.parent / "suite.json"

        return SuiteConfig.deserialize(suite_path)

    @staticmethod
    def deserialize(suite_path: Path) -> "SuiteConfig":
        with open(suite_path, "r") as file:
            data = json.load(file)
        return SuiteConfig(**data)

    @staticmethod
    def get_data_paths(suite_path: Path | str) -> List[str]:
        return glob.glob(f"{suite_path}/**/data.json", recursive=True)

    def challenge_from_datum(self, file_datum: list[dict[str, Any]]) -> "ChallengeData":
        same_task_data = {
            "name": self.prefix,
            "dependencies": self.dependencies,
            "category": self.shared_category,
            "task": self.task,
            "cutoff": self.cutoff,
        }

        # if the SuiteConfig does not yet have info or ground, we use the info and ground from the data.json
        if not self.info:
            same_task_data["info"] = {
                datum["name"]: datum["info"] for datum in file_datum
            }
        else:
            same_task_data["info"] = self.info

        if not self.ground:
            same_task_data["ground"] = {
                datum["name"]: datum["ground"] for datum in file_datum
            }
        else:
            same_task_data["ground"] = self.ground

        return ChallengeData(**same_task_data)

    def challenge_from_test_data(self, data: dict[str, Any]) -> "ChallengeData":
        same_task_data = {
            "name": data["name"],
            "dependencies": data["dependencies"],
            "category": data["category"],
            "info": data["info"],
            "ground": data["ground"],
        }

        if self.same_task:
            same_task_data["category"].extend(self.shared_category)
            same_task_data["task"] = self.task
            same_task_data["cutoff"] = self.cutoff
        else:
            same_task_data["task"] = data["task"]
            same_task_data["cutoff"] = data["cutoff"]

        return ChallengeData(**same_task_data)
