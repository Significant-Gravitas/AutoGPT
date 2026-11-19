"""Hand a webhook event to the tasks that watch it (the ingress side).

Called by the webhook ingress after the provider's manager has checked the
delivery. For each open task of the webhook's owner that watches it and whose
event filter takes the event, a trimmed copy goes on the task's queue and a
waiting task becomes due now, so the due-task sweep starts it with every check
a scheduled start passes. Nothing runs here: the ingress only writes.

A paused task ignores events; a task with ``MAX_PENDING_EVENTS`` already
waiting drops the new one and says so on its activity.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import prisma.models
from prisma.enums import TaskActivityKind, TaskStatus
from prisma.types import TaskUpdateManyMutationInput

from backend.blocks import get_block
from backend.data.task_record import ActivityInput, add_activity
from backend.data.task_schedule_meta import (
    MAX_PENDING_EVENTS,
    TaskWatch,
    WatchEvent,
    read_schedule_meta,
    with_schedule_meta,
)
from backend.data.task_watch_event import trim_event
from backend.data.task_watch_lookup import watching_task_rows

logger = logging.getLogger(__name__)

DROPPED_TEXT = f"Skipped an event: {MAX_PENDING_EVENTS} were already waiting to run"
# Another write to the task between read and write: read it again.
_WRITE_TRIES = 3


async def deliver_webhook_event(
    user_id: str,
    webhook_id: str,
    event_type: str,
    payload: Any,
    now: datetime | None = None,
) -> int:
    """Queue the event on each of the owner's tasks that watch the webhook
    and take this event. Returns how many queued it."""
    rows = await watching_task_rows(user_id, webhook_id)
    if not rows:
        return 0
    now = now or datetime.now(UTC)
    event = trim_event(event_type, payload, now)
    queued = 0
    for row in rows:
        watch = _watch_of(row)
        if watch is None or not takes_event(watch, event_type):
            continue
        if await queue_event(row, event, now):
            queued += 1
    return queued


def takes_event(watch: TaskWatch, event_type: str) -> bool:
    """Whether the watch's event filter takes ``event_type``, read the way a
    trigger node reads its own (``Block.is_triggered_by_event_type``)."""
    block = get_block(watch.trigger_block_id)
    if block is None or block.webhook_config is None:
        return False
    try:
        return block.is_triggered_by_event_type(watch.trigger_config, event_type)
    except ValueError:
        return False


async def queue_event(
    row: prisma.models.Task, event: WatchEvent, now: datetime
) -> bool:
    """Put ``event`` on the task's queue and, while it waits, make it due
    now. Conditional on the row being as read, so a run claimed meanwhile
    does not lose the event. False when it ignored or dropped the event."""
    current: prisma.models.Task | None = row
    for _ in range(_WRITE_TRIES):
        if current is None or _watch_of(current) is None:
            return False
        meta = read_schedule_meta(current.metadata)
        assert meta.repeat is not None
        if meta.repeat.paused:
            return False
        if len(meta.events) >= MAX_PENDING_EVENTS:
            await _note_dropped(current)
            return False
        if await _write(current, _queued(current, event, now)):
            return True
        current = await prisma.models.Task.prisma().find_first(
            where={"id": row.id, "userId": row.userId, "isArchived": False}
        )
    logger.warning(f"Could not queue an event on task {row.id}: it kept changing")
    return False


def _queued(
    row: prisma.models.Task, event: WatchEvent, now: datetime
) -> TaskUpdateManyMutationInput:
    meta = read_schedule_meta(row.metadata)
    waits = row.status == TaskStatus.TODO and row.startAt is None
    update: dict[str, Any] = {"events": [*meta.events, event]}
    if waits:
        update.update({"due_at": now, "missed": False})
    data: TaskUpdateManyMutationInput = {
        "metadata": with_schedule_meta(row.metadata, meta.model_copy(update=update))
    }
    if waits:
        data["startAt"] = now
        data["statusText"] = None
    return data


async def _write(row: prisma.models.Task, data: TaskUpdateManyMutationInput) -> bool:
    moved = await prisma.models.Task.prisma().update_many(
        where={
            "id": row.id,
            "userId": row.userId,
            "isArchived": False,
            "updatedAt": row.updatedAt,
        },
        data=data,
    )
    return moved > 0


def _watch_of(row: prisma.models.Task) -> TaskWatch | None:
    repeat = read_schedule_meta(row.metadata).repeat
    return repeat.watch if repeat else None


async def _note_dropped(row: prisma.models.Task) -> None:
    try:
        await add_activity(
            row.id,
            row.userId,
            ActivityInput(kind=TaskActivityKind.STATUS, text=DROPPED_TEXT),
        )
    except Exception as e:
        logger.warning(f"Could not note a dropped event on task {row.id}: {e}")
