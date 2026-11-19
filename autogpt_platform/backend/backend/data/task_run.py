"""Run a task from its own page: claim it for a new run, undo a claim whose
turn never started, and settle it when the run's chat ends.

A run started this way has no chat waiting on it, so nothing else closes the
task: the executor settles it at the end of every turn in the run's chat
(``backend.copilot.task_run_settle``).
"""

import re
from datetime import UTC, datetime
from typing import Literal

import prisma.models
from prisma import Json
from prisma.enums import TaskStatus
from prisma.types import TaskUpdateManyMutationInput
from pydantic import BaseModel

from backend.data.task import LINE_MAX_CHARS, OPEN_STATUSES, Task, get_task
from backend.data.task_record import EXCERPT_MAX_CHARS, redact, settle_work_tasks
from backend.data.task_rhythm import is_repeating
from backend.data.task_schedule_meta import (
    MAX_PENDING_EVENTS,
    TaskScheduleMeta,
    WatchEvent,
    read_schedule_meta,
    with_schedule_meta,
)
from backend.data.task_text import clip
from backend.util.exceptions import ConflictError, NotFoundError

# A task can be run again from any of these; one that is being worked on or
# waits on the user is refused.
RUNNABLE_STATUSES = [
    TaskStatus.TODO,
    TaskStatus.FAILED,
    TaskStatus.DONE,
    TaskStatus.CANCELLED,
]
# Running one of these again is another try at the same work, so it counts
# one more attempt.
RETRY_STATUSES = [TaskStatus.FAILED, TaskStatus.CANCELLED]

# How a turn in the run's chat ended.
TurnEnding = Literal["completed", "failed", "timed_out", "cancelled"]

_FAILURE_BY_ENDING: dict[TurnEnding, str] = {
    "failed": "The work stopped before it finished.",
    "timed_out": "The work ran out of time.",
}
NO_RESULT_REASON = "The work ended without a result."
# The reason shows on the task and in its "Failed: <reason>" activity line,
# which is capped at one line; ``redact`` cuts past its excerpt length.
_REASON_MAX_CHARS = min(LINE_MAX_CHARS - len("Failed: "), EXCERPT_MAX_CHARS)
_REPLY_SENTENCES = 2
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A ``finish_task`` summary that opens by saying the work could not be done.
_BLOCKED_OPENING = re.compile(
    r"^(blocked\b|(i )?(could not|couldn't|cannot|can't|was unable to|"
    r"am unable to|unable to)\b)",
    re.IGNORECASE,
)


class TaskRunClaim(BaseModel):
    """A task just moved to IN_PROGRESS for a new run, and how it was before."""

    before: Task
    task: Task
    retry: bool
    # Started by the due-task sweep at its start time, not by the user.
    scheduled: bool = False
    # The event this run of a watching task is for, taken off its queue.
    event: WatchEvent | None = None


async def claim_task_run(task_id: str, user_id: str) -> TaskRunClaim:
    """Move the task to IN_PROGRESS for a new run. A failed or cancelled task
    counts one more attempt; a suggested one becomes the user's; a scheduled or
    missed one starts now, its start time gone. Conditional on the status read,
    so two clicks at once start one run."""
    before = await get_task(task_id, user_id)
    if before is None:
        raise NotFoundError(f"Task {task_id} not found")
    if before.status not in RUNNABLE_STATUSES:
        raise ConflictError(f"'{before.title}' is already being worked on.")
    retry = before.status in RETRY_STATUSES
    data: TaskUpdateManyMutationInput = {
        "status": TaskStatus.IN_PROGRESS,
        "isSuggested": False,
        # The last run's outcome goes: this run leaves its own.
        "result": None,
        "failureReason": None,
        "statusText": None,
        "nextStep": None,
        "startedAt": datetime.now(UTC),
        "completedAt": None,
        "startAt": None,
    }
    if before.missed_start:
        data["metadata"] = await _without_missed_start(task_id)
    if retry:
        data["attempt"] = before.attempt + 1
    moved = await prisma.models.Task.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "isArchived": False,
            "status": before.status,
        },
        data=data,
    )
    if moved == 0:
        raise ConflictError(f"'{before.title}' changed while starting. Try again.")
    task = await get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")
    return TaskRunClaim(before=before, task=task, retry=retry)


async def _without_missed_start(task_id: str) -> Json:
    """The schedule part cleared; a repeating task keeps its rhythm and the
    chat its runs report to."""
    row = await prisma.models.Task.prisma().find_unique(where={"id": task_id})
    metadata = row.metadata if row else None
    kept = read_schedule_meta(metadata)
    meta = (
        TaskScheduleMeta(
            repeat=kept.repeat,
            origin_session_id=kept.origin_session_id,
            events=kept.events,
        )
        if kept.repeat
        else TaskScheduleMeta()
    )
    return with_schedule_meta(metadata, meta)


async def undo_task_run(claim: TaskRunClaim, user_id: str) -> None:
    """Put a claimed task back as it was: its run never started, and the
    event it took goes back to the front of the queue."""
    before = claim.before
    data: TaskUpdateManyMutationInput = await _event_put_back(claim)
    await prisma.models.Task.prisma().update_many(
        where={"id": before.id, "userId": user_id, "status": TaskStatus.IN_PROGRESS},
        data={
            **data,
            "status": before.status,
            "isSuggested": before.is_suggested,
            "result": before.result,
            "failureReason": before.failure_reason,
            "statusText": before.status_text,
            "nextStep": before.next_step,
            "attempt": before.attempt,
            "startedAt": before.started_at,
            "completedAt": before.completed_at,
            "workSessionId": before.work_session_id,
            "startAt": before.start_at,
        },
    )


async def _event_put_back(claim: TaskRunClaim) -> TaskUpdateManyMutationInput:
    if claim.event is None:
        return {}
    row = await prisma.models.Task.prisma().find_first(
        where={"id": claim.task.id, "userId": claim.task.user_id}
    )
    if row is None:
        return {}
    meta = read_schedule_meta(row.metadata)
    events = [claim.event, *meta.events][:MAX_PENDING_EVENTS]
    return {
        "metadata": with_schedule_meta(
            row.metadata, meta.model_copy(update={"events": events})
        )
    }


async def settle_task_run(
    task_id: str,
    session_id: str,
    user_id: str,
    ending: TurnEnding,
    last_reply: str | None = None,
) -> TaskStatus | None:
    """Close the run when a turn in its chat ends: DONE when the worker left
    a result (``finish_task``), FAILED when the turn failed or ended without
    one, CANCELLED when it was stopped. A turn that ended without a result
    fails with what the worker last said (``last_reply``) as the reason, and
    so does one whose result only says it was blocked (``is_blocked_result``).
    A task waiting on the user, one with subtasks still open, or one whose
    work moved to another chat stays as it is. Returns the status it moved
    to, or None."""
    row = await prisma.models.Task.prisma().find_first(
        where={
            "id": task_id,
            "userId": user_id,
            "isArchived": False,
            "workSessionId": session_id,
            "status": {"in": [TaskStatus.TODO, TaskStatus.IN_PROGRESS]},
        },
        include={"Subtasks": True, "Artifacts": True, "Evidence": True},
    )
    if row is None:
        return None
    if row.status == TaskStatus.TODO and is_repeating(row):
        # Waiting for its next run: this chat's run already ended.
        return None
    status, reason = _settled_status(row, ending, last_reply)
    if status is None:
        return None
    await settle_work_tasks(session_id, user_id, status, None, reason)
    return status


def _settled_status(
    row: prisma.models.Task, ending: TurnEnding, last_reply: str | None = None
) -> tuple[TaskStatus | None, str | None]:
    if ending == "cancelled":
        return TaskStatus.CANCELLED, None
    if ending != "completed":
        return TaskStatus.FAILED, _FAILURE_BY_ENDING[ending]
    if any(child.status in OPEN_STATUSES for child in row.Subtasks or []):
        # Its subtasks report back to this chat, whose next turn settles it.
        return None, None
    if row.result and is_blocked_result(row):
        return TaskStatus.FAILED, reply_reason(row.result) or NO_RESULT_REASON
    if row.result:
        return TaskStatus.DONE, None
    return TaskStatus.FAILED, reply_reason(last_reply) or NO_RESULT_REASON


def is_blocked_result(row: prisma.models.Task) -> bool:
    """A result that opens with "Blocked" / "Could not" / "Unable to" while
    the task saved no file and recorded no proof: the worker reported that it
    could not do the work, so the task did not get done."""
    opening = (row.result or "").lstrip(" *_#>`'\"").strip()
    made = [a for a in row.Artifacts or [] if not a.isMissing]
    return bool(_BLOCKED_OPENING.match(opening)) and not made and not row.Evidence


def reply_reason(reply: str | None) -> str | None:
    """The worker's last words as a failure reason: its first sentence or
    two, as one line within the cap, with anything shaped like a key taken
    out. None when it said nothing."""
    text = " ".join((reply or "").replace("**", "").replace("`", "").split())
    opening = " ".join(_SENTENCE_END.split(text)[:_REPLY_SENTENCES])
    return redact(clip(opening, _REASON_MAX_CHARS))
