from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, constr, validator


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
    description: constr(regex=r"^Tests if the agent can.*")
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
    case_sensitive: Optional[bool] = True
    eval: Eval


class Category(str, Enum):
    DATA = "data"
    GENERALIST = "general"
    CODING = "coding"
    SCRAPE_SYNTHESIZE = "scrape_synthesize"
    GAIA_1 = "GAIA_1"
    GAIA_2 = "GAIA_2"
    GAIA_3 = "GAIA_3"


class ChallengeData(BaseModel):
    eval_id: str = ""
    name: str
    category: List[Category]
    task: str
    dependencies: List[str]
    cutoff: int
    ground: Ground | Dict[str, Ground]
    info: Info | Dict[str, Info]
    metadata: Optional[Dict[str, Any]] = None

    spec_file: Path | None = Field(None, exclude=True)
