"""Live task updates over the user notification bus.

The websocket server forwards every ``NotificationEvent`` on this bus to the
user's browser as ``{"method": "notification", "data": payload}``.
"""

import logging
from typing import Any

from backend.api.model import TaskEventNotificationPayload
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)

logger = logging.getLogger(__name__)

_bus: AsyncRedisNotificationEventBus | None = None


def _get_bus() -> AsyncRedisNotificationEventBus:
    global _bus
    if _bus is None:
        _bus = AsyncRedisNotificationEventBus()
    return _bus


def _dump(value: Any) -> dict | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return None


async def publish_contract_update(user_id: str, update: Any) -> None:
    """Publish a ``contract_db.ContractUpdate``: the small contract patch, then
    the status event (needs you, completed) its patch caused, if any."""
    task = update.task
    activity = _without_content(update.activity)
    if update.patches:
        await publish_task_event(
            user_id,
            "task_contract",
            None,
            activity,
            task_id=task.id,
            task_key=task.key,
            title=task.title,
            contract_patch={
                "version": update.version,
                "state": task.state.value,
                "changes": [p.model_dump(mode="json") for p in update.patches],
            },
        )
    if update.status_event:
        await publish_task_event(user_id, update.status_event, task, activity, task.id)


def _without_content(activity: Any | None) -> dict | None:
    """Evidence content can be large; it is fetched on demand, never pushed."""
    dumped = _dump(activity)
    if not dumped or "content" not in (dumped.get("data") or {}):
        return dumped
    return {**dumped, "data": {**dumped["data"], "content": None}}


async def publish_task_event(
    user_id: str,
    event: str,
    task: Any | None,
    activity: Any | None = None,
    task_id: str | None = None,
    task_key: str | None = None,
    title: str | None = None,
    contract_patch: dict | None = None,
) -> None:
    """Publish a ``task_event`` to the user's notification stream. Never raises."""
    try:
        task_json = _dump(task)
        activity_json = _dump(activity)
        resolved_id = (
            task_id
            or (task_json or {}).get("id")
            or (activity_json or {}).get("task_id")
        )
        if not resolved_id:
            logger.warning("Skipping task event %s without a task id", event)
            return
        payload = TaskEventNotificationPayload(
            type="task_event",
            event=event,
            task_id=resolved_id,
            task_key=task_key or (task_json or {}).get("key"),
            title=title or (task_json or {}).get("title"),
            task=task_json,
            activity=activity_json,
            contract_patch=contract_patch,
        )
        await _get_bus().publish(NotificationEvent(user_id=user_id, payload=payload))
    except Exception:
        logger.warning(
            "Failed to publish task event %s for task %s",
            event,
            task_id,
            exc_info=True,
        )
