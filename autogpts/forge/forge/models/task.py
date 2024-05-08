from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .artifact import Artifact
from .utils import Pagination


class StepOutput(BaseModel):
    pass


class TaskRequestBody(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        description="Input prompt for the task.",
        example="Write the words you receive to the file 'output.txt'.",
    )
    additional_input: Optional[dict] = None


class Task(TaskRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task.",
        example="50da533e-3904-4401-8a07-c49adf88b5eb",
    )
    artifacts: Optional[List[Artifact]] = Field(
        [],
        description="A list of artifacts that the task has produced.",
        example=[
            "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
            "ab7b4091-2560-4692-a4fe-d831ea3ca7d6",
        ],
    )


class StepRequestBody(BaseModel):
    name: Optional[str] = Field(
        None, description="The name of the task step.", example="Write to file"
    )
    input: Optional[str] = Field(
        None,
        description="Input prompt for the step.",
        example="Washington",
    )
    additional_input: Optional[dict] = None


class Status(Enum):
    created = "created"
    running = "running"
    completed = "completed"


class Step(StepRequestBody):
    created_at: datetime = Field(
        ...,
        description="The creation datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    modified_at: datetime = Field(
        ...,
        description="The modification datetime of the task.",
        example="2023-01-01T00:00:00Z",
        json_encoders={datetime: lambda v: v.isoformat()},
    )
    task_id: str = Field(
        ...,
        description="The ID of the task this step belongs to.",
        example="50da533e-3904-4401-8a07-c49adf88b5eb",
    )
    step_id: str = Field(
        ...,
        description="The ID of the task step.",
        example="6bb1801a-fd80-45e8-899a-4dd723cc602e",
    )
    name: Optional[str] = Field(
        None, description="The name of the task step.", example="Write to file"
    )
    status: Status = Field(
        ..., description="The status of the task step.", example="created"
    )
    output: Optional[str] = Field(
        None,
        description="Output of the task step.",
        example="I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')",
    )
    additional_output: Optional[dict] = None
    artifacts: Optional[List[Artifact]] = Field(
        [], description="A list of artifacts that the step has produced."
    )
    is_last: bool = Field(
        ..., description="Whether this is the last step in the task.", example=True
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
