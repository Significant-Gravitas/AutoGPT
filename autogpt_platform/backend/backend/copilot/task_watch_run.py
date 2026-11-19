"""A run of a watching task: what it is told about the event it is for, and
the line its start leaves on the task (the run list reads ``trigger``)."""

from typing import Any

from backend.data.task import LINE_MAX_CHARS
from backend.data.task_run import TaskRunClaim
from backend.data.task_text import clip
from backend.data.task_watch_event import what_happened

TRIGGER_KEY = "trigger"


def event_brief(claim: TaskRunClaim) -> str:
    """The task's brief, with the event it runs for after it as data."""
    brief = claim.task.description or claim.task.title
    if claim.event is None:
        return brief
    return f"{brief}\n\n{what_happened(claim.event)}"


def started_line(claim: TaskRunClaim, default: str) -> str:
    """What the start says on the task: what happened, for an event."""
    if claim.event is None:
        return default
    return clip(f"Started for: {claim.event.summary}", LINE_MAX_CHARS) or default


def started_data(claim: TaskRunClaim) -> dict[str, Any]:
    return {TRIGGER_KEY: claim.event.summary} if claim.event else {}
