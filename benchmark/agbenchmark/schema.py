from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    pass


class TaskRequestBody(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        description="Input prompt for the task.",
        example="Write the words you receive to the file 'output.txt'.",
    )
    additional_input: Optional[TaskInput] = {}


class TaskEvalRequestBody(TaskRequestBody):
    eval_id: str
