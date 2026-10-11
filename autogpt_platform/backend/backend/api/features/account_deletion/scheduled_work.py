"""Stop a closing account's scheduled and running work."""

import asyncio
import logging
from collections.abc import Awaitable
from typing import TypeVar

from backend.data.execution import ExecutionStatus, get_graph_executions
from backend.executor.scheduler import SchedulerClient
from backend.executor.utils import stop_graph_execution
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# How long each stop waits for its run to confirm it ended. The cancel goes
# out before the wait, so this bounds the confirmation, not the stop itself:
# queued and incomplete runs are marked terminated on the first poll, and a
# running one that needs longer has already been told to stop. Every stop runs
# at once with cascade off (a child run belongs to the same user, so it is in
# the list itself), so the whole step takes about this long however many runs
# there are and however deeply they nest.
STOP_WAIT_TIMEOUT_SECONDS = 5.0

# The most any one call may take. These run inside the deletion request, and
# the scheduler RPC client would otherwise wait up to its 300 s default.
STEP_TIMEOUT_SECONDS = 10.0

_UNFINISHED_STATUSES = [
    ExecutionStatus.INCOMPLETE,
    ExecutionStatus.QUEUED,
    ExecutionStatus.RUNNING,
]


async def stop_user_work(user_id: str) -> None:
    """Remove every schedule the user owns, then stop their unfinished runs.

    Schedules go first so none can fire again after the runs are listed; a
    fire already under way when its schedule is removed can still add a run.
    Best effort: each call that fails is logged and the rest still run, and
    this never raises, so the account can still be closed.
    """
    if not user_id:
        # An empty id would turn the run listing into "every user's runs".
        logger.error("Account deletion stopped no work: no user id was given")
        return
    await _remove_schedules(user_id)
    await _stop_unfinished_runs(user_id)


async def _remove_schedules(user_id: str) -> None:
    try:
        scheduler = get_scheduler_client()
    except Exception as e:
        _log_failure("reach the scheduler", user_id, e)
        return
    await asyncio.gather(
        _remove_listed_schedules(scheduler, user_id),
        _attempt(
            "remove the morning briefing",
            user_id,
            scheduler.remove_morning_briefing_schedule(user_id=user_id),
        ),
        _attempt(
            "remove the nightly batch",
            user_id,
            scheduler.delete_nightly_batch_schedule(user_id=user_id),
        ),
        _attempt(
            "remove the community rebuild",
            user_id,
            scheduler.delete_community_rebuild_schedule(user_id=user_id),
        ),
    )


async def _remove_listed_schedules(scheduler: SchedulerClient, user_id: str) -> None:
    """Graph and copilot-turn schedules, paused ones (a fired expert's) included."""
    schedules = await _attempt(
        "list schedules",
        user_id,
        scheduler.get_execution_schedules(user_id=user_id, include_paused=True),
    )
    # By job id, not schedule_id: a legacy job has no schedule_id in its args.
    await asyncio.gather(
        *(
            _attempt(
                f"delete schedule #{schedule.id}",
                user_id,
                scheduler.delete_schedule(schedule_id=schedule.id, user_id=user_id),
            )
            for schedule in schedules or []
        )
    )


async def _stop_unfinished_runs(user_id: str) -> None:
    runs = await _attempt(
        "list unfinished runs",
        user_id,
        get_graph_executions(user_id=user_id, statuses=_UNFINISHED_STATUSES),
    )
    await asyncio.gather(
        *(
            _attempt(f"stop run #{run.id}", user_id, _stop_run(user_id, run.id))
            for run in runs or []
        )
    )


async def _stop_run(user_id: str, run_id: str) -> None:
    try:
        await stop_graph_execution(
            user_id=user_id,
            graph_exec_id=run_id,
            wait_timeout=STOP_WAIT_TIMEOUT_SECONDS,
            cascade=False,
        )
    except TimeoutError:
        logger.warning(
            f"Account deletion sent a stop to run #{run_id} of user {user_id}, "
            f"which had not stopped after {STOP_WAIT_TIMEOUT_SECONDS:.0f}s"
        )


async def _attempt(what: str, user_id: str, step: Awaitable[T]) -> T | None:
    """Await one call, bounded by STEP_TIMEOUT_SECONDS; log it if it fails."""
    try:
        return await asyncio.wait_for(step, timeout=STEP_TIMEOUT_SECONDS)
    except NotFoundError:
        logger.info(
            f"Account deletion did not need to {what} for user {user_id}: already gone"
        )
    except Exception as e:
        _log_failure(what, user_id, e)
    return None


def _log_failure(what: str, user_id: str, error: Exception) -> None:
    logger.error(
        f"Account deletion could not {what} for user {user_id}: "
        f"{type(error).__name__}: {error}"
    )
