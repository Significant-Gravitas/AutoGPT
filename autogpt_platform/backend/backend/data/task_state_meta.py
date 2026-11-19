"""What a task keeps about its run and its review in ``Task.metadata``, beside
the schedule part (``task_schedule_meta``), with no column of its own.

- ``run_queued``: the run's first turn waits for a free slot (past the user's
  running cap); cleared when that turn starts. It names the run (its start
  time), so a mark an earlier run left behind never counts.
- ``review``: the user accepted the result, or asked for changes. It belongs
  to the run that made the result, so it reads as absent once a later run
  has started.

Each key is read on its own, so a malformed one never costs the others.
"""

from datetime import datetime
from typing import Any, Literal

from prisma import Json
from pydantic import BaseModel, Field, ValidationError

RUN_QUEUED_KEY = "run_queued"
REVIEW_KEY = "review"
REVIEW_NOTE_MAX_CHARS = 1_000

ReviewState = Literal["accepted", "changes_requested"]


class TaskReview(BaseModel):
    """The user's word on a task's result."""

    state: ReviewState
    note: str | None = Field(
        default=None, description="What to change, when changes were asked for."
    )
    at: datetime


def read_run_queued(metadata: Any, started_at: datetime | None) -> bool:
    """Whether the run that started at ``started_at`` still waits for a slot.
    The mark names its run, so one left by an earlier run never counts."""
    return started_at is not None and _raw(metadata).get(RUN_QUEUED_KEY) == run_mark(
        started_at
    )


def run_mark(started_at: datetime) -> str:
    return started_at.isoformat()


def read_review(metadata: Any, started_at: datetime | None) -> TaskReview | None:
    """The review of the task's latest result; None when there is none or it
    was given before the latest run started (it reviewed an older result)."""
    raw = _raw(metadata).get(REVIEW_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        review = TaskReview.model_validate(raw)
    except ValidationError:
        return None
    if started_at is not None and review.at < started_at:
        return None
    return review


def with_state(metadata: Any, key: str, value: dict[str, Any] | str | None) -> Json:
    """The task's metadata with ``key`` set to ``value`` (None removes the
    key), every other key kept."""
    raw = {name: kept for name, kept in _raw(metadata).items() if name != key}
    if value is not None:
        raw[key] = value
    return Json(raw)


def _raw(metadata: Any) -> dict[str, Any]:
    return metadata if isinstance(metadata, dict) else {}
