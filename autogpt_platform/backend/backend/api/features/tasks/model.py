from prisma.enums import TaskStatus
from pydantic import BaseModel, Field

from backend.data.task import (
    DESCRIPTION_MAX_CHARS,
    LINE_MAX_CHARS,
    TITLE_MAX_CHARS,
    Task,
)

# Ids are UUIDs; the bound only keeps junk out of the lookup.
ID_MAX_CHARS = 64


class TaskCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=TITLE_MAX_CHARS)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX_CHARS)
    owner_expert_id: str | None = Field(
        default=None,
        max_length=ID_MAX_CHARS,
        description="The expert who does the work; omit for Otto.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=ID_MAX_CHARS,
        description="A chat of the caller's to attach the task to.",
    )
    goal: str | None = Field(default=None, max_length=LINE_MAX_CHARS)
    done_when: str | None = Field(default=None, max_length=LINE_MAX_CHARS)


class TaskUpdateRequest(BaseModel):
    """Fields left out, or sent as null, stay as they are."""

    title: str | None = Field(default=None, min_length=1, max_length=TITLE_MAX_CHARS)
    description: str | None = Field(
        default=None,
        max_length=DESCRIPTION_MAX_CHARS,
        description="An empty string clears the description.",
    )
    status: TaskStatus | None = None
    owner_expert_id: str | None = Field(default=None, max_length=ID_MAX_CHARS)
    assign_to_otto: bool = Field(
        default=False, description="Clear the owner expert so Otto owns the task."
    )
    goal: str | None = Field(default=None, max_length=LINE_MAX_CHARS)
    done_when: str | None = Field(default=None, max_length=LINE_MAX_CHARS)


class TaskListResponse(BaseModel):
    tasks: list[Task]
