from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskRequestBody(BaseModel):
    input: str = Field(
        min_length=1,
        description="Input prompt for the task.",
        examples=["Write the words you receive to the file 'output.txt'."],
    )
    additional_input: Optional[dict[str, Any]] = Field(default_factory=dict)


class TaskEvalRequestBody(TaskRequestBody):
    eval_id: str
