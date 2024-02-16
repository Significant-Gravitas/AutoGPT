from enum import Enum
from typing import Literal

from pydantic import BaseModel


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


class Category(str, Enum):
    GENERALIST = "general"
    DATA = "data"
    CODING = "coding"
    SCRAPE_SYNTHESIZE = "scrape_synthesize"
    WEB = "web"
    GAIA_1 = "GAIA_1"
    GAIA_2 = "GAIA_2"
    GAIA_3 = "GAIA_3"


class EvalResult(BaseModel):
    result: str
    result_source: Literal["step_output"] | str
    score: float
    passed: bool
