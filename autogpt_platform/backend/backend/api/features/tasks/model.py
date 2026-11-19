from datetime import datetime

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
INSTRUCTIONS_MAX_CHARS = 2_000


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
    """Fields left out, or sent as null, stay as they are; ``start_at`` sent
    as null removes the start time."""

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
    start_at: datetime | None = Field(
        default=None,
        description="A future time for a TODO task to start by itself; null "
        "removes the start time. A time without an offset is UTC.",
    )


class TaskRunRequest(BaseModel):
    instructions: str | None = Field(
        default=None,
        max_length=INSTRUCTIONS_MAX_CHARS,
        description="What to do differently this run; kept on the task's activity.",
    )


class TaskCancelRequest(BaseModel):
    this_run_only: bool = Field(
        default=False,
        description="For a repeating task: stop only the run going now, and "
        "keep the rhythm.",
    )


class TaskListResponse(BaseModel):
    tasks: list[Task]
