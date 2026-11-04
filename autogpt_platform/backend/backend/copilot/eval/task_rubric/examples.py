"""The 50 labelled messages of the task rubric, read from the engineering docs."""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

REPO_ROOT = Path(__file__).resolve().parents[6]
EXAMPLES_PATH = REPO_ROOT / "docs" / "engineering" / "task-rubric-examples.json"


class Verdict(str, Enum):
    TASK = "Task"
    PARENT_WITH_SUBTASKS = "Parent+sub-tasks"
    NOT_A_TASK = "Not a task"
    UPDATE_EXISTING = "Update existing task"
    ROUTINE = "Routine"
    ASK_THEN_TASK = "Ask then task"


class RubricExample(BaseModel):
    id: int
    input: str
    verdict: Verdict
    rule: str
    why: str


def load_examples(path: Path = EXAMPLES_PATH) -> list[RubricExample]:
    return TypeAdapter(list[RubricExample]).validate_python(
        json.loads(path.read_text())
    )
