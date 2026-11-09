from prisma.enums import TaskStatus
from pydantic import BaseModel, Field

from backend.data.task import DESCRIPTION_MAX_CHARS, TITLE_MAX_CHARS, Task


class TaskCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=TITLE_MAX_CHARS)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_CHARS)
    owner_expert_id: str | None = Field(
        default=None, description="The expert who does the work; omit for Otto."
    )
    session_id: str | None = Field(
        default=None, description="A chat of the caller's to attach the task to."
    )


class TaskUpdateRequest(BaseModel):
    """Fields left out, or sent as null, stay as they are."""

    title: str | None = Field(default=None, min_length=1, max_length=TITLE_MAX_CHARS)
    description: str | None = Field(
        default=None,
        max_length=DESCRIPTION_MAX_CHARS,
        description="An empty string clears the description.",
    )
    status: TaskStatus | None = None
    owner_expert_id: str | None = None
    assign_to_otto: bool = Field(
        default=False, description="Clear the owner expert so Otto owns the task."
    )


class TaskListResponse(BaseModel):
    tasks: list[Task]
