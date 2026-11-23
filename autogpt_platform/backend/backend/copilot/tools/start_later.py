"""``start_task`` with a start time: the task is saved now and starts later.

Nothing runs and nothing calls the Scheduler service when the task is saved,
so the chat is free at once. The task row holds the start time and the
due-task sweep (``backend.copilot.task_due``) starts it then, in the
background, with the result coming back to the chat that asked.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from prisma.enums import TaskActivityKind, TaskActor, TaskStatus
from pydantic import BaseModel

from backend.copilot import turn_end
from backend.copilot.model import ChatSession
from backend.copilot.start_time import describe_start, resolve_start_time, user_timezone
from backend.data.db_accessors import task_record_db, task_schedule_db
from backend.data.task import Task
from backend.data.task_record import ActivityInput
from backend.data.task_rhythm import RhythmPlan, plan_rhythm
from backend.util.settings import Settings

from .models import ErrorResponse, ToolResponseBase
from .task_models import TaskInfo, TaskStartedResponse

logger = logging.getLogger(__name__)


class StartPlan(BaseModel):
    """When a task should start: ``at`` None means now. With ``rhythm`` it
    repeats, ``at`` being its first run."""

    at: datetime | None = None
    timezone: str = "UTC"
    timezone_saved: bool = False
    rhythm: RhythmPlan | None = None


async def plan_start(
    user_id: str,
    start_in_seconds: int | None,
    start_at: str | None,
    repeat_crons: list[str] | None = None,
    seed: str = "",
) -> StartPlan:
    """Read the start, or the rhythm, the model asked for. Raises ValueError,
    written for the model, when it is malformed."""
    if repeat_crons:
        if start_in_seconds is not None or start_at:
            raise ValueError(
                "Give repeat_crons or a start time, not both: a repeating "
                "task first runs at its first cron time."
            )
        timezone, saved = await user_timezone(user_id)
        rhythm = plan_rhythm(repeat_crons, timezone, datetime.now(UTC), seed)
        return StartPlan(
            at=rhythm.first_at, timezone=timezone, timezone_saved=saved, rhythm=rhythm
        )
    if start_in_seconds is None and not start_at:
        return StartPlan()
    timezone, saved = await user_timezone(user_id)
    at = resolve_start_time(start_in_seconds, start_at, timezone, datetime.now(UTC))
    return StartPlan(at=at, timezone=timezone, timezone_saved=saved)


def waiting_to_start(task: Task) -> bool:
    """A task that has not started: scheduled, or past a start it missed."""
    return task.status == TaskStatus.TODO and (
        task.start_at is not None or task.missed_start
    )


async def schedule_task(
    user_id: str,
    session: ChatSession,
    plan: StartPlan,
    waiting: Task | None,
    create: Callable[[str], Awaitable[Task]],
) -> ToolResponseBase:
    """Save the task with its start time (``create`` makes it, with the given
    first activity line), or give ``waiting``, the same task asked for
    earlier, the new time. Then wrap the turn up."""
    assert plan.at is not None
    when = describe_start(plan.at, plan.timezone)
    rhythm = plan.rhythm
    if await task_record_db().work_tasks(session.session_id, user_id):
        return _error(
            "A start time is for a task asked for in a chat, not for a part of "
            "a task. Start this part now.",
            session,
        )
    if waiting is None:
        refusal = await _cap_refusal(user_id)
        if refusal:
            return _error(refusal, session)
        task = await create(
            f"Repeats: {rhythm.repeat.label}. First run {when}"
            if rhythm
            else f"Scheduled to start {when}"
        )
    else:
        task = waiting
        line = (
            f"Now repeats: {rhythm.repeat.label}. Next run {when}"
            if rhythm
            else f"Start moved to {when}"
        )
        await _note(task, user_id, session, line, plan.at)
    rhythm_kw = {"repeat": rhythm.repeat} if rhythm else {}
    task = await task_schedule_db().set_task_start(
        task.id, user_id, plan.at, session.session_id, **rhythm_kw
    )
    turn_end.request_wrap_up(session.session_id)
    message = (
        _repeat_message(task, when, plan, rhythm)
        if rhythm
        else _scheduled_message(task, when, plan, waiting is not None)
    )
    return TaskStartedResponse(
        message=message,
        task=TaskInfo.from_task(task),
        in_background=True,
        start_at=plan.at,
        session_id=session.session_id,
    )


async def _cap_refusal(user_id: str) -> str | None:
    limit = Settings().config.max_scheduled_tasks_per_user
    if await task_schedule_db().count_scheduled_tasks(user_id) < limit:
        return None
    return (
        f"The user already has {limit} tasks waiting to start, the most "
        "allowed. Nothing was scheduled. Tell them to cancel one or start one "
        "now to free a slot."
    )


def _scheduled_message(task: Task, when: str, plan: StartPlan, moved: bool) -> str:
    zone_note = (
        ""
        if plan.timezone_saved
        else " The user has no time zone saved, so say the time is in UTC."
    )
    verb = "Moved the start of" if moved else "Scheduled"
    return (
        f"{verb} '{task.title}' to {when}. Nothing runs now: at that time it "
        "starts in the background and the result comes back to this chat. "
        "Tell the user in one short line when it starts, with that exact "
        f"time.{zone_note} Do none of the task here."
    )


def _repeat_message(task: Task, when: str, plan: StartPlan, rhythm: RhythmPlan) -> str:
    zone_note = (
        ""
        if plan.timezone_saved
        else " The user has no time zone saved, so say the times are in UTC."
    )
    cost_note = (
        " It runs more often than every 15 minutes and every run uses their "
        "allowance: say so in that line and offer a slower rhythm."
        if rhythm.is_costly
        else ""
    )
    return (
        f"'{task.title}' now repeats ({rhythm.repeat.label}); the first run "
        f"is {when}. Nothing runs now: each run starts in the background and "
        "its result comes back to this chat. It keeps repeating until the "
        "user pauses or stops it on the task. Tell the user in one short line "
        f"how often it runs and when the first run is.{zone_note}{cost_note} "
        "Do none of the task here."
    )


async def _note(
    task: Task, user_id: str, session: ChatSession, text: str, at: datetime
) -> None:
    """Add a line to the task's activity; the schedule matters more."""
    try:
        await task_record_db().add_activity(
            task.id,
            user_id,
            ActivityInput(
                kind=TaskActivityKind.STATUS,
                text=text,
                actor_type=TaskActor.EXPERT if session.expert_id else TaskActor.OTTO,
                actor_expert_id=session.expert_id,
                chat_session_id=session.session_id,
                is_milestone=True,
                data={"start_at": at.isoformat()},
            ),
        )
    except Exception as e:
        logger.warning("Could not note activity on task %s: %s", task.id, e)


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(
        message=message, error="start_task_failed", session_id=session.session_id
    )
