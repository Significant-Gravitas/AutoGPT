"""What a scheduled task keeps about its start in ``Task.metadata``.

``Task.startAt`` is the next time the sweep tries to start it. The metadata
holds the rest under one key: the time it was first due (a start pushed back
by a missing plan stays late against that time), the chat that asked (its
result goes back there), whether it missed its start and, for a repeating
task, its rhythm (``TaskRepeat``). A task started by an event is a repeating
task whose next start is set by the event instead of a cron: its rhythm holds
what it watches (``TaskWatch``) and the schedule the events still to run.
"""

from datetime import datetime
from typing import Any, Literal

from prisma import Json
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEDULE_KEY = "schedule"

RunEnd = Literal["done", "failed", "cancelled"]


# Events waiting for their run, oldest first; past this one is dropped.
MAX_PENDING_EVENTS = 5


class TaskWatch(BaseModel):
    """What a task started by an event watches: one webhook of the user's,
    and the trigger block whose event filter picks the events it runs for.
    Holds no credential, secret or account name."""

    webhook_id: str
    provider: str
    trigger_block_id: str
    trigger_config: dict[str, Any] = Field(
        default_factory=dict,
        description="The trigger block's inputs (resource, event filter), "
        "without credentials.",
    )
    line: str = Field(description='What it watches, in words: "PR opened on a/b".')
    url: str | None = Field(
        default=None,
        description="Where to send events, for a webhook the user calls.",
    )


class WatchEvent(BaseModel):
    """One event a watching task runs for, trimmed to the fields that matter.
    Outside content: a run reads it as data, never as instructions."""

    received_at: datetime
    event_type: str
    summary: str = Field(description='A few words: "PR #12 opened: Fix login".')
    payload_text: str = Field(description="The trimmed payload as JSON text.")


class TaskRepeat(BaseModel):
    """A task that runs again and again: after each run it goes back to
    waiting, with ``Task.startAt`` at the next fire time of ``crons``, or, for
    a task that ``watch``es, with no start until the next event."""

    crons: list[str] = Field(
        default_factory=list,
        description="5-field crons read in ``timezone``, the dialect routines use.",
    )
    timezone: str = "UTC"
    label: str = Field(description="How often it runs, in words.")
    paused: bool = Field(
        default=False, description="Paused: it keeps its rhythm but does not start."
    )
    failed_in_a_row: int = Field(
        default=0, description="Runs that failed one after another, up to now."
    )
    last_run: RunEnd | None = Field(default=None, description="How the last run ended.")
    watch: TaskWatch | None = Field(
        default=None, description="Set when an event starts each run, not a cron."
    )


class TaskScheduleMeta(BaseModel):
    due_at: datetime | None = None
    origin_session_id: str | None = None
    missed: bool = False
    repeat: TaskRepeat | None = None
    events: list[WatchEvent] = Field(
        default_factory=list, description="A watching task's events still to run."
    )


class _TaskMetadata(BaseModel):
    """The task's metadata, other keys kept as they are."""

    model_config = ConfigDict(extra="allow")

    schedule: TaskScheduleMeta = TaskScheduleMeta()


def read_schedule_meta(metadata: Any) -> TaskScheduleMeta:
    """The schedule part of a task's metadata; empty when absent or malformed."""
    return _parse(metadata).schedule


def with_schedule_meta(metadata: Any, meta: TaskScheduleMeta) -> Json:
    """The task's metadata with its schedule part replaced, other keys kept."""
    parsed = _parse(metadata)
    parsed.schedule = meta
    return Json(parsed.model_dump(mode="json"))


def _parse(metadata: Any) -> _TaskMetadata:
    try:
        return _TaskMetadata.model_validate(metadata or {})
    except ValidationError:
        return _TaskMetadata()
