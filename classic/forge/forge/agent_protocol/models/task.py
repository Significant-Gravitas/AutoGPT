from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .artifact import Artifact
from .pagination import Pagination


class TaskRequestBody(BaseModel):
    input: str = Field(
        min_length=1,
        description="Input prompt for the task.",
        examples=["Write the words you receive to the file 'output.txt'."],
    )
    additional_input: dict[str, Any] = Field(default_factory=dict)


class Task(TaskRequestBody):
    created_at: datetime = Field(
        description="The creation datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
    )
    modified_at: datetime = Field(
        description="The modification datetime of the task.",
        examples=["2023-01-01T00:00:00Z"],
    )
    task_id: str = Field(
        description="The ID of the task.",
        examples=["50da533e-3904-4401-8a07-c49adf88b5eb"],
    )
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="A list of artifacts that the task has produced.",
        examples=[
            "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
            "ab7b4091-2560-4692-a4fe-d831ea3ca7d6",
        ],
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )


class StepRequestBody(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="The name of the task step.",
        examples=["Write to file"],
    )
    input: str = Field(
        description="Input prompt for the step.", examples=["Washington"]
    )
    additional_input: dict[str, Any] = Field(default_factory=dict)


class StepStatus(Enum):
    created = "created"
    running = "running"
    completed = "completed"


class Step(StepRequestBody):
    created_at: datetime = Field(
        description="The creation datetime of the task.",
        examples=[
            "2023-01-01T00:00:00Z",
        ],
    )
    modified_at: datetime = Field(
        description="The modification datetime of the task.",
        examples=[
            "2023-01-01T00:00:00Z",
        ],
    )
    task_id: str = Field(
        description="The ID of the task this step belongs to.",
        examples=[
            "50da533e-3904-4401-8a07-c49adf88b5eb",
        ],
    )
    step_id: str = Field(
        description="The ID of the task step.",
        examples=[
            "6bb1801a-fd80-45e8-899a-4dd723cc602e",
        ],
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the task step.",
        examples=["Write to file"],
    )
    status: StepStatus = Field(
        description="The status of the task step.", examples=["created"]
    )
    output: Optional[str] = Field(
        default=None,
        description="Output of the task step.",
        examples=[
            "I am going to use the write_to_file command and write Washington "
            "to a file called output.txt <write_to_file('output.txt', 'Washington')"
        ],
    )
    additional_output: Optional[dict[str, Any]] = None
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="A list of artifacts that the step has produced.",
    )
    is_last: bool = Field(
        description="Whether this is the last step in the task.", examples=[True]
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )


class TaskListResponse(BaseModel):
    tasks: Optional[List[Task]] = None
    pagination: Optional[Pagination] = None


class TaskStepsListResponse(BaseModel):
    steps: Optional[List[Step]] = None
    pagination: Optional[Pagination] = None


class TaskArtifactsListResponse(BaseModel):
    artifacts: Optional[List[Artifact]] = None
    pagination: Optional[Pagination] = None
