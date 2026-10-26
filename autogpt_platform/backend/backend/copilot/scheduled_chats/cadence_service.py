"""Run a scheduled chat less often (SECRT-2879).

The note a chat leaves when its owner's usage limit ran out offers this beside
Upgrade. It reads how often the chat runs, offers the cadences below that
(``cadence``), and applies the one the owner picks: a routine through
``enable_routine``, the path every change to a routine's cadence takes, and a
follow-up in the scheduler, in place, so it keeps its id.

Runs in the REST API, which has both the database and the scheduler.
"""

from datetime import datetime, timezone

from prisma.models import ScheduledChatStreak
from pydantic import BaseModel

from backend.api.features.experts import routines
from backend.api.features.experts.models import ExpertRoutine
from backend.copilot.scheduled_chats.cadence import describe, less_often
from backend.copilot.scheduled_chats.notices import schedule_title
from backend.data.user import get_user_by_id
from backend.executor.scheduler import CopilotTurnJobInfo
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError
from backend.util.timezone_utils import get_user_timezone_or_utc


class CadenceOption(BaseModel):
    cron: str
    # "Every day at 09:00", on the chat's own clock.
    label: str


class ScheduledChatCadence(BaseModel):
    schedule_key: str
    title: str
    # How often it runs now, in words, when that can be said.
    current: str | None
    # The zone its times are in.
    timezone: str
    # Least drastic first; empty when it already runs once a week or less.
    options: list[CadenceOption]


class CadenceNotFound(Exception):
    """Nothing here to run less often: not the caller's, gone, switched off,
    or a one-off."""


class CadenceNotOffered(ValueError):
    """Not one of the cadences offered for this chat."""


class ScheduledChatPaused(Exception):
    """Paused after failing the same way: resuming it comes first."""


class _Schedule(BaseModel):
    key: str
    title: str
    crons: list[str]
    timezone: str
    # Set for a routine, whose row owns its crons.
    routine: ExpertRoutine | None = None


async def get_cadence(user_id: str, schedule_key: str) -> ScheduledChatCadence:
    return _cadence(await _schedule(user_id, schedule_key))


async def set_cadence(
    user_id: str, schedule_key: str, cron: str
) -> ScheduledChatCadence:
    """Switch the chat to *cron*, one of the cadences it was offered."""
    schedule = await _schedule(user_id, schedule_key)
    offered = less_often(schedule.crons, schedule.timezone, datetime.now(timezone.utc))
    if cron not in {rung.cron for rung in offered}:
        raise CadenceNotOffered(cron)
    if schedule.routine is not None:
        try:
            await routines.enable_routine(
                user_id, schedule.routine.expert_id, schedule.routine.id, crons=[cron]
            )
        except routines.RoutineNotFoundError as e:
            raise CadenceNotFound(schedule_key) from e
    else:
        try:
            await get_scheduler_client().update_copilot_turn_cron(
                schedule_key, user_id=user_id, cron=cron
            )
        except NotFoundError as e:
            # Deleted since it was read.
            raise CadenceNotFound(schedule_key) from e
    return _cadence(schedule.model_copy(update={"crons": [cron]}))


def _cadence(schedule: _Schedule) -> ScheduledChatCadence:
    now = datetime.now(timezone.utc)
    return ScheduledChatCadence(
        schedule_key=schedule.key,
        title=schedule.title,
        current=describe(schedule.crons, schedule.timezone, now),
        timezone=schedule.timezone,
        options=[
            CadenceOption(cron=rung.cron, label=rung.label)
            for rung in less_often(schedule.crons, schedule.timezone, now)
        ],
    )


async def _schedule(user_id: str, schedule_key: str) -> _Schedule:
    """The caller's scheduled chat behind *schedule_key*: a routine's id, or a
    follow-up's schedule id."""
    streak = await ScheduledChatStreak.prisma().find_unique(
        where={"userId_scheduleKey": {"userId": user_id, "scheduleKey": schedule_key}}
    )
    if streak is not None and streak.pausedAt is not None:
        raise ScheduledChatPaused(schedule_key)
    routine = await routines.get_routine(schedule_key)
    if routine is not None:
        return await _routine_schedule(user_id, routine)
    return await _follow_up_schedule(user_id, schedule_key)


async def _routine_schedule(user_id: str, routine: ExpertRoutine) -> _Schedule:
    # ``get_routine`` is unscoped; this is the routines' own ownership rule.
    owned = await routines.list_routines(user_id, routine.expert_id)
    if not any(row.id == routine.id for row in owned):
        raise CadenceNotFound(routine.id)
    if not routine.enabled or not routine.crons:
        raise CadenceNotFound(routine.id)
    # The zone ``enable_routine`` will put the new cadence in.
    user = await get_user_by_id(user_id)
    return _Schedule(
        key=routine.id,
        title=routine.title,
        crons=list(routine.crons),
        timezone=get_user_timezone_or_utc(user.timezone if user else None),
        routine=routine,
    )


async def _follow_up_schedule(user_id: str, schedule_key: str) -> _Schedule:
    jobs = await get_scheduler_client().get_execution_schedules(
        user_id=user_id, kind="copilot_turn"
    )
    job = next(
        (
            job
            for job in jobs
            if isinstance(job, CopilotTurnJobInfo) and job.id == schedule_key
        ),
        None,
    )
    if job is None or job.cron is None or job.routine_id is not None:
        raise CadenceNotFound(schedule_key)
    return _Schedule(
        key=job.id,
        title=schedule_title(routine_title=None, message=job.message),
        crons=[job.cron],
        # The scheduler puts the new cadence in this zone too.
        timezone=job.user_timezone or job.timezone,
    )
