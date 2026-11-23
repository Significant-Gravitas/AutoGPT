"""A start time or a rhythm given when a task is created (``POST /tasks``).

Both are checked before anything is written: a start must be in the future,
a rhythm reads the way repeating tasks read it (``task_rhythm.plan_rhythm``:
5-field crons, every minute at most often), and the two do not go together.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException
from prisma.enums import TaskActivityKind, TaskActor
from pydantic import BaseModel
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from backend.copilot.start_time import describe_start, user_timezone
from backend.data.task import Task
from backend.data.task_record import ActivityInput, add_activity
from backend.data.task_rhythm import RhythmPlan, plan_rhythm
from backend.data.task_schedule import set_task_start

from .model import TaskCreateRequest


class NewStart(BaseModel):
    """When a new task first starts, and its rhythm when it repeats."""

    at: datetime
    timezone: str
    rhythm: RhythmPlan | None = None


async def plan_new_start(user_id: str, body: TaskCreateRequest) -> NewStart | None:
    """The start the request asks for, or None to leave the task unscheduled.
    Raises a 422 written for the user when it cannot be used."""
    if body.start_at is not None and body.repeat is not None:
        raise _refused("Give a start time or a repeat, not both.")
    if body.repeat is not None:
        timezone = body.repeat.timezone or (await user_timezone(user_id))[0]
        _check_zone(timezone)
        try:
            rhythm = plan_rhythm(
                body.repeat.crons,
                timezone,
                datetime.now(UTC),
                seed=f"{user_id}:{body.title}",
            )
        except ValueError as e:
            raise _refused(str(e).replace("repeat_crons", "A repeat")) from e
        return NewStart(at=rhythm.first_at, timezone=timezone, rhythm=rhythm)
    if body.start_at is None:
        return None
    at = body.start_at if body.start_at.tzinfo else body.start_at.replace(tzinfo=UTC)
    if at <= datetime.now(UTC):
        raise _refused("Choose a start time in the future.")
    timezone, _ = await user_timezone(user_id)
    return NewStart(at=at, timezone=timezone)


async def schedule_new_task(task: Task, user_id: str, start: NewStart) -> Task:
    """Give the just-created task its start (and rhythm), with one line on its
    activity saying when it runs."""
    repeat = start.rhythm.repeat if start.rhythm else None
    task = await set_task_start(task.id, user_id, start.at, None, repeat=repeat)
    when = describe_start(start.at, start.timezone)
    line = (
        f"Repeats: {repeat.label}. First run {when}"
        if repeat
        else f"Scheduled to start {when}"
    )
    await add_activity(
        task.id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.STATUS,
            text=line,
            actor_type=TaskActor.USER,
            is_milestone=True,
            data={"start_at": start.at.isoformat()},
        ),
    )
    return task


def _check_zone(timezone: str) -> None:
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise _refused(f"Unknown time zone: {timezone}.") from e


def _refused(detail: str) -> HTTPException:
    return HTTPException(status_code=HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)
