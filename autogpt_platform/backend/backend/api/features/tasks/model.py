from datetime import datetime
from typing import Annotated, Literal

from prisma.enums import TaskStatus
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.data.task import (
    DESCRIPTION_MAX_CHARS,
    LINE_MAX_CHARS,
    TITLE_MAX_CHARS,
    Task,
)
from backend.data.task_rhythm import MAX_REPEAT_CRONS
from backend.data.task_state_meta import REVIEW_NOTE_MAX_CHARS

# Ids are UUIDs; the bound only keeps junk out of the lookup.
ID_MAX_CHARS = 64
INSTRUCTIONS_MAX_CHARS = 2_000
# A 5-field cron is short; the bound only keeps junk out.
CRON_MAX_CHARS = 100


class TaskRepeatRequest(BaseModel):
    """A rhythm, as repeating tasks take it: 5-field crons, every minute at
    most often."""

    model_config = ConfigDict(extra="forbid")

    crons: list[Annotated[str, StringConstraints(max_length=CRON_MAX_CHARS)]] = Field(
        min_length=1, max_length=MAX_REPEAT_CRONS
    )
    timezone: str | None = Field(
        default=None,
        max_length=64,
        description="IANA time zone the crons are read in; omit for the user's.",
    )


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    start_at: datetime | None = Field(
        default=None,
        description="A future time for the task to start by itself. A time "
        "without an offset is UTC. Not with repeat.",
    )
    repeat: TaskRepeatRequest | None = Field(
        default=None,
        description="Run the task again and again on this rhythm; its first "
        "run is the next fire time. Not with start_at.",
    )


class TaskUpdateRequest(BaseModel):
    """Fields left out, or sent as null, stay as they are; ``start_at`` sent
    as null removes the start time."""

    model_config = ConfigDict(extra="forbid")

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


class TaskReviewRequest(BaseModel):
    decision: Literal["accept", "request_changes"]
    note: str | None = Field(
        default=None,
        max_length=REVIEW_NOTE_MAX_CHARS,
        description="What to change; required to ask for changes.",
    )


class TaskListResponse(BaseModel):
    tasks: list[Task]
    has_more: bool = Field(default=False, description="More tasks follow this page.")
    next_offset: int | None = Field(
        default=None, description="The offset of the next page, when there is one."
    )
