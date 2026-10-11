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


async def pause_jobs(user_id: str, schedule_ids: list[str]) -> bool:
    """Pause every job, and say whether that worked. On a failure the jobs
    this call paused are resumed again. A job deleted since fires no more,
    so it is skipped; with no live job left there is nothing to pause."""
    scheduler = get_scheduler_client()
    paused: list[str] = []
    live = False
    for schedule_id in schedule_ids:
        try:
            # False: already paused, by its owner, and it stays that way.
            if await scheduler.pause_schedule(schedule_id, user_id=user_id):
                paused.append(schedule_id)
            live = True
        except NotFoundError:
            continue
        except Exception:
            logger.warning(
                "Could not pause scheduled chat job %s", schedule_id, exc_info=True
            )
            await _put_back(scheduler.resume_schedule, user_id, paused)
            return False
    return live


async def resume_jobs(user_id: str, schedule_ids: list[str]) -> None:
    """Resume every job, or raise with the ones this call resumed paused
    again. A job deleted since the pause has nothing to resume."""
    scheduler = get_scheduler_client()
    resumed: list[str] = []
    for schedule_id in schedule_ids:
        try:
            if await scheduler.resume_schedule(schedule_id, user_id=user_id):
                resumed.append(schedule_id)
        except NotFoundError:
            continue
        except Exception:
            await _put_back(scheduler.pause_schedule, user_id, resumed)
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
