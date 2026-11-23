"""Give a task a start time, change it or take it away; and pause, resume or
stop a repeating task, or one that watches for an event.

Only a TODO task waits for a start time: a task being worked on, or one that
is settled, is refused. Every write is scoped to the owning user. The sweep
that starts due tasks lives in ``task_due``; what a repeating task does when a
run ends lives in ``task_rhythm``.
"""

from datetime import UTC, datetime
from typing import Any

import prisma.models
from prisma.enums import TaskStatus
from prisma.types import TaskUpdateManyMutationInput

from backend.data.task import OPEN_STATUSES, TASK_GONE, Task, get_task
from backend.data.task_rhythm import (
    STATUS_BY_RUN_END,
    next_run_at,
    next_watch_run_at,
    rearmed,
)
from backend.data.task_schedule_meta import (
    TaskRepeat,
    TaskScheduleMeta,
    read_schedule_meta,
    with_schedule_meta,
)
from backend.data.task_watch_setup import release_watch_webhook
from backend.util.exceptions import ConflictError, NotFoundError


async def set_task_start(
    task_id: str,
    user_id: str,
    start_at: datetime | None,
    origin_session_id: str | None = None,
    repeat: TaskRepeat | None = None,
) -> Task:
    """Schedule the TODO task to start at ``start_at``, or unschedule it with
    None. Either way it is no longer missed and its waiting note goes.
    ``origin_session_id`` names the chat its result goes back to; None keeps
    the one it had. ``repeat`` gives it a rhythm; None keeps the one it had."""
    row = await _live_row(task_id, user_id)
    if row.status != TaskStatus.TODO:
        raise ConflictError(f"'{row.title}' has already started.")
    meta = read_schedule_meta(row.metadata)
    meta = TaskScheduleMeta(
        due_at=start_at,
        origin_session_id=origin_session_id or meta.origin_session_id,
        repeat=repeat or meta.repeat,
        events=meta.events,
    )
    moved = await prisma.models.Task.prisma().update_many(
        where={"id": task_id, "userId": user_id, "status": TaskStatus.TODO},
        data={
            "startAt": start_at,
            "statusText": None,
            "metadata": with_schedule_meta(row.metadata, meta),
        },
    )
    if moved == 0:
        raise ConflictError(f"'{row.title}' changed meanwhile. Try again.")
    return await _task(task_id, user_id)


async def count_scheduled_tasks(user_id: str) -> int:
    """How many of the user's tasks wait for a start time."""
    return await prisma.models.Task.prisma().count(
        where={
            "userId": user_id,
            "isArchived": False,
            "isSuggested": False,
            "status": TaskStatus.TODO,
            "startAt": {"not": None},
        }
    )


async def cancel_scheduled_tasks(user_id: str) -> int:
    """Cancel every task of the user's that waits for a start time or an
    event and stop every repeating one: the account is closing, or closed. A
    task that missed its start never starts by itself, so it is left. Returns
    how many were cancelled."""
    watching = await _stop_all_repeats(user_id)
    return await prisma.models.Task.prisma().update_many(
        where={
            "userId": user_id,
            "status": TaskStatus.TODO,
            "OR": [{"startAt": {"not": None}}, {"id": {"in": watching}}],
        },
        data={
            "status": TaskStatus.CANCELLED,
            "startAt": None,
            "completedAt": datetime.now(UTC),
        },
    )


async def pause_task_repeat(task_id: str, user_id: str) -> Task:
    """Pause a repeating task: it keeps its rhythm and starts no more runs.
    A run going now finishes; the task then waits, paused."""
    row, repeat = await _repeating_row(task_id, user_id)
    data = _with_repeat(row, repeat.model_copy(update={"paused": True}))
    if row.status == TaskStatus.TODO:
        data.update({"startAt": None, "statusText": None})
    return await _write(row, data)


async def resume_task_repeat(task_id: str, user_id: str) -> Task:
    """Resume a paused repeating task at its next fire time (a watching task:
    now for an event still waiting, else at the next event); the failed runs
    counted so far are forgotten."""
    row, repeat = await _repeating_row(task_id, user_id)
    repeat = repeat.model_copy(update={"paused": False, "failed_in_a_row": 0})
    if row.status != TaskStatus.TODO:
        return await _write(row, _with_repeat(row, repeat))
    now = datetime.now(UTC)
    if repeat.watch is not None:
        next_at = next_watch_run_at(read_schedule_meta(row.metadata), now)
    else:
        next_at = next_run_at(repeat, now)
        if next_at is None:
            raise ConflictError(f"'{row.title}' has no run left in its rhythm.")
    data = _with_repeat(row, repeat, due_at=next_at)
    data.update({"startAt": next_at, "statusText": None})
    return await _write(row, data)


async def stop_task_repeat(task_id: str, user_id: str) -> Task:
    """Stop a task repeating, or watching. A run going now settles as a
    one-off does; a task waiting for its next run settles as its last run
    ended, or, if it never ran, keeps its next start as a one-off (a watching
    task that never ran is cancelled). A watched webhook nothing else uses is
    let go of."""
    row, repeat = await _repeating_row(task_id, user_id)
    data = _with_repeat(row, None)
    ended = repeat.last_run or ("cancelled" if repeat.watch else None)
    if row.status == TaskStatus.TODO and ended is not None:
        data.update(
            {
                "status": STATUS_BY_RUN_END[ended],
                "startAt": None,
                "statusText": None,
            }
        )
    task = await _write(row, data)
    if repeat.watch is not None:
        await release_watch_webhook(user_id, repeat.watch)
    return task


async def clear_task_repeat(task_id: str, user_id: str) -> bool:
    """Take the rhythm off a task that was cancelled or settled by hand, so
    running it again later runs it once. False when it had none."""
    row = await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    repeat = read_schedule_meta(row.metadata).repeat if row else None
    if row is None or repeat is None:
        return False
    await prisma.models.Task.prisma().update_many(
        where={"id": task_id, "userId": user_id}, data=_with_repeat(row, None)
    )
    if repeat.watch is not None:
        await release_watch_webhook(user_id, repeat.watch)
    return True


async def end_repeat_run(task_id: str, user_id: str) -> Task | None:
    """Stop only the run going now of a repeating task: it waits for its next
    run, as when a run ends. None when no run of a repeating task is going."""
    row = await prisma.models.Task.prisma().find_first(
        where={
            "id": task_id,
            "userId": user_id,
            "isArchived": False,
            "status": {"in": [TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_INPUT]},
        }
    )
    if row is None:
        return None
    now = datetime.now(UTC)
    rearm = rearmed(row, "cancelled", now)
    if rearm is None:
        return None
    moved = await prisma.models.Task.prisma().update_many(
        where={"id": task_id, "userId": user_id, "status": row.status},
        data={
            "status": TaskStatus.TODO,
            "startAt": rearm.start_at,
            "statusText": rearm.paused_because,
            "nextStep": None,
            "completedAt": now,
            "metadata": with_schedule_meta(row.metadata, rearm.meta),
        },
    )
    return await get_task(task_id, user_id) if moved else None


async def _stop_all_repeats(user_id: str) -> list[str]:
    """Stop every repeating task of the user's; the ids of the ones that
    watched, whose webhooks are let go of."""
    rows = await prisma.models.Task.prisma().find_many(
        where={"userId": user_id, "status": {"in": OPEN_STATUSES}}
    )
    watching: list[str] = []
    for row in rows:
        repeat = read_schedule_meta(row.metadata).repeat
        if repeat is None:
            continue
        await prisma.models.Task.prisma().update_many(
            where={"id": row.id, "userId": user_id},
            data=_with_repeat(row, None),
        )
        if repeat.watch is not None:
            watching.append(row.id)
            await release_watch_webhook(user_id, repeat.watch)
    return watching


async def _repeating_row(
    task_id: str, user_id: str
) -> tuple[prisma.models.Task, TaskRepeat]:
    row = await _live_row(task_id, user_id)
    repeat = read_schedule_meta(row.metadata).repeat
    if repeat is None:
        raise ConflictError(f"'{row.title}' does not repeat.")
    if row.status not in OPEN_STATUSES:
        raise ConflictError(f"'{row.title}' is no longer open.")
    return row, repeat


def _with_repeat(
    row: prisma.models.Task,
    repeat: TaskRepeat | None,
    due_at: datetime | None = None,
) -> TaskUpdateManyMutationInput:
    meta = read_schedule_meta(row.metadata)
    update: dict[str, Any] = {"repeat": repeat, "missed": False}
    if repeat is None:
        update["events"] = []
    if due_at is not None:
        update["due_at"] = due_at
    return {
        "metadata": with_schedule_meta(row.metadata, meta.model_copy(update=update))
    }


async def _write(row: prisma.models.Task, data: TaskUpdateManyMutationInput) -> Task:
    """Write ``data`` if the task is still as it was read."""
    moved = await prisma.models.Task.prisma().update_many(
        where={"id": row.id, "userId": row.userId, "status": row.status},
        data=data,
    )
    if moved == 0:
        raise ConflictError(f"'{row.title}' changed meanwhile. Try again.")
    return await _task(row.id, row.userId)


async def _live_row(task_id: str, user_id: str) -> prisma.models.Task:
    row = await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id, "isArchived": False}
    )
    if row is None:
        raise NotFoundError(TASK_GONE)
    return row


async def _task(task_id: str, user_id: str) -> Task:
    task = await get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(TASK_GONE)
    return task
