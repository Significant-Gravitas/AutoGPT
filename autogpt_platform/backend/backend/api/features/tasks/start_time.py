"""A start time the user sets on a task (``PATCH /tasks/{id}``).

Only a task that has not started takes a start time: one being worked on or
settled is refused before anything is written. Setting, moving or removing it
also clears a missed start (``set_task_start``).
"""

from datetime import UTC, datetime

from fastapi import HTTPException
from prisma.enums import TaskStatus
from starlette.status import HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_CONTENT

from backend.copilot.start_time import describe_start, user_timezone
from backend.data.task import TASK_GONE, Task

from .model import TaskUpdateRequest


def requested_start(before: Task | None, body: TaskUpdateRequest) -> datetime | None:
    """The start time the request asks for (None removes it), checked against
    the task as it is. Raises HTTPException when it cannot take one."""
    if before is None:
        raise HTTPException(status_code=404, detail=TASK_GONE)
    status = body.status or before.status
    if status != TaskStatus.TODO:
        raise HTTPException(
            status_code=HTTP_409_CONFLICT,
            detail="Only a task that has not started can get a start time.",
        )
    if body.start_at is None:
        return None
    start_at = body.start_at
    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=UTC)
    if start_at <= datetime.now(UTC):
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Choose a start time in the future.",
        )
    return start_at


async def start_change_line(before: Task | None, after: Task) -> str | None:
    """What the user just did to the start time, as one activity line."""
    if before is None or before.start_at == after.start_at:
        return None
    if after.start_at is None:
        return "You removed the start time"
    timezone, _ = await user_timezone(after.user_id)
    when = describe_start(after.start_at, timezone)
    if before.start_at is None:
        return f"You set it to start {when}"
    return f"You moved the start to {when}"
