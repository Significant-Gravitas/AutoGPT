"""Pausing and resuming a scheduled chat's jobs, all of them or none
(SECRT-2879).

A chat that reads as paused must not fire, and one that reads as running must
not be stuck: a routine with two fire times that paused only one would be
both. So a pause or a resume that fails partway puts back what it changed.
"""

import logging
from typing import Any, Awaitable, Callable

from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


async def pause_jobs(user_id: str, schedule_ids: list[str]) -> list[str] | None:
    """Pause every job that is due to fire. Returns the jobs this call paused,
    the only ones a resume may restart: a job its owner paused, or one deleted
    since, is left as it is. None when the scheduler refused partway, after
    resuming what this call had paused, and the job it failed on if that one
    was running: a pause whose answer was lost may still have landed."""
    if not schedule_ids:
        return []
    scheduler = get_scheduler_client()
    try:
        running = {
            job.id
            for job in await scheduler.get_execution_schedules(
                user_id=user_id, kind="copilot_turn"
            )
        }
    except Exception:
        logger.warning("Could not read the scheduled chat jobs", exc_info=True)
        return None
    paused: list[str] = []
    for schedule_id in schedule_ids:
        try:
            # False: already paused by its owner, and not ours to restart.
            if await scheduler.pause_schedule(schedule_id, user_id=user_id):
                paused.append(schedule_id)
        except NotFoundError:
            continue
        except Exception:
            logger.warning(
                "Could not pause scheduled chat job %s", schedule_id, exc_info=True
            )
            # Resuming one whose pause never landed changes nothing; one its
            # owner had paused was not running, and stays as it is.
            uncertain = [schedule_id] if schedule_id in running else []
            await unpause(user_id, paused + uncertain)
            return None
    return paused


async def unpause(user_id: str, schedule_ids: list[str]) -> None:
    """Resume jobs that ``pause_jobs`` paused, best effort."""
    await _put_back(get_scheduler_client().resume_schedule, user_id, schedule_ids)


async def resume_jobs(user_id: str, schedule_ids: list[str]) -> None:
    """Resume every job, or raise with the ones this call resumed paused
    again, the one it failed on included: a resume whose answer was lost may
    have landed. A job deleted since the pause has nothing to resume."""
    scheduler = get_scheduler_client()
    resumed: list[str] = []
    for schedule_id in schedule_ids:
        try:
            if await scheduler.resume_schedule(schedule_id, user_id=user_id):
                resumed.append(schedule_id)
        except NotFoundError:
            continue
        except Exception:
            # Pausing one whose resume never landed changes nothing: it is
            # still paused.
            await _put_back(scheduler.pause_schedule, user_id, [*resumed, schedule_id])
            raise


async def jobs_paused(user_id: str, schedule_ids: list[str]) -> bool:
    """Whether none of these jobs is due to fire. Paused jobs are left out of
    the scheduler's list. Unknown counts as paused."""
    try:
        active = await get_scheduler_client().get_execution_schedules(
            user_id=user_id, kind="copilot_turn"
        )
    except Exception:
        logger.warning("Could not read the scheduled chat jobs", exc_info=True)
        return True
    return not {job.id for job in active} & set(schedule_ids)


async def _put_back(
    undo: Callable[..., Awaitable[Any]], user_id: str, schedule_ids: list[str]
) -> None:
    for schedule_id in schedule_ids:
        try:
            await undo(schedule_id, user_id=user_id)
        except Exception:
            logger.warning(
                "Could not put scheduled chat job %s back", schedule_id, exc_info=True
            )
