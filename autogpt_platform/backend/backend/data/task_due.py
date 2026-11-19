"""The due-task sweep's reads and writes (``backend.copilot.task_due``).

Every write is conditional on the task still being TODO with the start time
the sweep read, so two sweeps, or a sweep and the user pressing Start now or
Cancel, never both act on one task: the second write finds nothing and the
caller skips the task.
"""

from datetime import UTC, datetime

import prisma.models
from prisma import Json
from prisma.enums import TaskStatus
from prisma.types import TaskUpdateManyMutationInput, TaskWhereInput
from pydantic import BaseModel

from backend.data.task import _WITH_SESSIONS, Task, get_task
from backend.data.task_run import TaskRunClaim
from backend.data.task_schedule_meta import (
    TaskRepeat,
    WatchEvent,
    read_schedule_meta,
    with_schedule_meta,
)

DUE_BATCH_MAX = 100


class DueTask(BaseModel):
    """A task whose start time has come."""

    task: Task
    start_at: datetime
    # When it was first due; later than ``start_at`` only after a hold.
    due_at: datetime
    origin_session_id: str | None = None
    repeat: TaskRepeat | None = None


async def list_due_tasks(now: datetime, limit: int) -> list[DueTask]:
    """Every user's tasks due by ``now``, oldest first. One indexed query
    (``status``, ``startAt``) that finds nothing when nothing is due."""
    rows = await prisma.models.Task.prisma().find_many(
        where={
            "status": TaskStatus.TODO,
            "startAt": {"lte": now},
            "isArchived": False,
            "isSuggested": False,
        },
        include=_WITH_SESSIONS,
        order={"startAt": "asc"},
        take=max(1, min(limit, DUE_BATCH_MAX)),
    )
    return [_due(row) for row in rows if row.startAt is not None]


async def get_scheduled_task(task_id: str, user_id: str) -> DueTask | None:
    """The user's task while it waits for its start time, for starting it
    early the way the sweep would; None for any other task."""
    row = await prisma.models.Task.prisma().find_first(
        where={
            "id": task_id,
            "userId": user_id,
            "status": TaskStatus.TODO,
            "startAt": {"not": None},
            "isArchived": False,
            "isSuggested": False,
        },
        include=_WITH_SESSIONS,
    )
    return _due(row) if row is not None else None


async def claim_due_task(
    task_id: str, user_id: str, start_at: datetime
) -> TaskRunClaim | None:
    """Move the due task to IN_PROGRESS for its run and clear its start time;
    None when it is no longer due at ``start_at`` (started, re-timed or
    cancelled meanwhile)."""
    before = await get_task(task_id, user_id)
    if before is None:
        return None
    where = _still_due(task_id, user_id, start_at)
    data: TaskUpdateManyMutationInput = {
        "status": TaskStatus.IN_PROGRESS,
        "startAt": None,
        "statusText": None,
        "nextStep": None,
        "result": None,
        "failureReason": None,
        "startedAt": datetime.now(UTC),
        "completedAt": None,
    }
    event = None
    if before.repeat and before.repeat.watch:
        event, metadata, updated_at = await _next_event(task_id, user_id)
        if metadata is not None:
            data["metadata"] = metadata
            # An event queued meanwhile changes the row: this claim then
            # misses, and the next sweep takes the task with both events.
            where["updatedAt"] = updated_at
    moved = await prisma.models.Task.prisma().update_many(where=where, data=data)
    if moved == 0:
        return None
    task = await get_task(task_id, user_id)
    if task is None:
        return None
    return TaskRunClaim(
        before=before, task=task, retry=False, scheduled=True, event=event
    )


async def _next_event(
    task_id: str, user_id: str
) -> tuple[WatchEvent | None, Json | None, datetime | None]:
    """A watching task's oldest waiting event, and its metadata without it."""
    row = await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    if row is None:
        return None, None, None
    meta = read_schedule_meta(row.metadata)
    if not meta.events:
        return None, None, None
    rest = meta.model_copy(update={"events": meta.events[1:]})
    return meta.events[0], with_schedule_meta(row.metadata, rest), row.updatedAt


async def mark_task_missed(
    task_id: str, user_id: str, start_at: datetime, status_text: str
) -> bool:
    """Too late to start: clear the start time, keep the task TODO and mark it
    missed, so the user chooses to start or cancel it."""
    return await _rewrite(
        task_id,
        user_id,
        start_at,
        next_start_at=None,
        status_text=status_text,
        missed=True,
    )


async def hold_due_task(
    task_id: str,
    user_id: str,
    expected_start_at: datetime | None,
    next_start_at: datetime,
    status_text: str | None,
    due_at: datetime,
) -> bool:
    """Try the task again at ``next_start_at``, saying why in ``status_text``.
    ``due_at`` is the time it counts as due for the lateness rule.
    ``expected_start_at`` None puts back a task whose start failed after its
    claim was undone."""
    return await _rewrite(
        task_id,
        user_id,
        expected_start_at,
        next_start_at=next_start_at,
        status_text=status_text,
        due_at=due_at,
    )


async def _rewrite(
    task_id: str,
    user_id: str,
    expected_start_at: datetime | None,
    *,
    next_start_at: datetime | None,
    status_text: str | None,
    missed: bool = False,
    due_at: datetime | None = None,
) -> bool:
    where = _still_due(task_id, user_id, expected_start_at)
    row = await prisma.models.Task.prisma().find_first(where=where)
    if row is None:
        return False
    meta = read_schedule_meta(row.metadata)
    meta = meta.model_copy(update={"missed": missed, "due_at": due_at or meta.due_at})
    moved = await prisma.models.Task.prisma().update_many(
        where=where,
        data={
            "startAt": next_start_at,
            "statusText": status_text,
            "metadata": with_schedule_meta(row.metadata, meta),
        },
    )
    return moved > 0


def _still_due(task_id: str, user_id: str, start_at: datetime | None) -> TaskWhereInput:
    return {
        "id": task_id,
        "userId": user_id,
        "status": TaskStatus.TODO,
        "startAt": start_at,
        "isArchived": False,
        "isSuggested": False,
    }


def _due(row: prisma.models.Task) -> DueTask:
    meta = read_schedule_meta(row.metadata)
    start_at = row.startAt or datetime.now(UTC)
    return DueTask(
        task=Task.from_db(row),
        start_at=start_at,
        due_at=meta.due_at or start_at,
        origin_session_id=meta.origin_session_id,
        repeat=meta.repeat,
    )
