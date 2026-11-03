"""Live task updates over the user notification bus (stub — replaced by dispatch work)."""

from typing import Any


async def publish_task_event(
    user_id: str,
    event: str,
    task: Any | None,
    activity: Any | None = None,
    task_id: str | None = None,
) -> None:
    return None
