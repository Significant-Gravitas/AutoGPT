"""What the dream-system cron bodies in ``executor/scheduler.py`` need: the
gate they ask before running, the run stamp and in-band-delete bookkeeping
they write, how they read and log a pass's outcome, and their jobs' args and
results.

``scheduler.py`` keeps the cron functions and RPC methods that APScheduler
and clients call by name. Everything here is plain async or pure code, and
the expert lifecycle lookup is handed in, so imports go scheduler → dream.
Registry calls run under the registry deadline and fail soft.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import ScopeRunKind

from .deadline import within_deadline
from .scheduling import DreamSystemJob, clear_registration_marker
from .scope_jobs import remove_scope_jobs, scope_label

if TYPE_CHECKING:
    from .nightly_batch import NightlyBatchResult

logger = logging.getLogger(__name__)

# ``(owner_user_id, expert_id) -> "active" | "archived" | ...``: the
# scheduler's expert lifecycle lookup.
ExpertStatusLookup = Callable[[str, str], Awaitable[str]]


async def memory_scope_may_fire(
    scope: MemoryScope, expert_status: ExpertStatusLookup
) -> bool:
    """Whether a dream-system cron of ``scope`` may run now.

    The registry must let it: an ACTIVE row, or an account with no row
    (crons from before the registry). An expert with no row was deleted, so
    its orphaned crons are removed. An expert scope's expert must also be
    active; archived, paused or unavailable only skips the run, since its
    crons are the lifecycle hooks' to manage. The whole check runs under the
    registry deadline, and a slow or failed answer skips the run.
    """
    try:
        return await within_deadline(_may_fire(scope, expert_status))
    except TimeoutError:
        logger.warning(f"Memory cron of {scope_label(scope)} skipped: gate timed out")
    except Exception:
        logger.warning(
            f"Memory cron of {scope_label(scope)} skipped: gate failed", exc_info=True
        )
    return False


async def record_scope_run(scope: MemoryScope, kind: ScopeRunKind) -> bool:
    """Stamp the time a cron body ran its pass without an error."""
    try:
        return await within_deadline(
            memory_schedule_db().record_scope_run(
                scope.owner_user_id, scope.scope_key, kind
            )
        )
    except Exception:
        logger.warning(
            f"Could not stamp the {kind} run of {scope_label(scope)}", exc_info=True
        )
        return False


async def forget_registration(scope: MemoryScope, job: DreamSystemJob) -> None:
    """After ``job`` was deleted outside the registry, drop its id from the
    row and its Redis marker so the next ensure registers it again."""
    await clear_registration_marker(scope, job.registration_key_prefix)
    try:
        await within_deadline(
            memory_schedule_db().forget_scope_job(
                scope.owner_user_id, scope.scope_key, job.job_id(scope)
            )
        )
    except Exception:
        logger.warning(
            f"Could not forget {job.name} of {scope_label(scope)}", exc_info=True
        )


def nightly_error_parts(result: "NightlyBatchResult") -> list[str]:
    """Every error a nightly result carries.

    ``run_nightly_batch_submit`` never raises: a submitter CRASH is captured
    in ``NightlyBatchResult.error``, while a submitter that ran but reported
    its own failure carries it on its sub-result (``dream.error`` /
    ``ratification.error``) with the top-level error left unset. A run is
    clean only when all of these are empty.
    """
    dream_error = result.dream.error if result.dream is not None else None
    ratification_error = (
        result.ratification.error if result.ratification is not None else None
    )
    return [
        part
        for part in (
            result.error,
            f"dream: {dream_error}" if dream_error else None,
            f"ratification: {ratification_error}" if ratification_error else None,
        )
        if part
    ]


def log_nightly_outcome(scope: MemoryScope, result: "NightlyBatchResult") -> None:
    label = f"{scope_label(scope)} (nightly {result.nightly_id})"
    if result.error:
        logger.warning(f"Nightly batch errored for {label}: {result.error}")
        return
    if result.skipped:
        logger.info(f"Nightly batch skipped for {label}: {result.skip_reason}")
        return
    dream, ratification = result.dream, result.ratification
    logger.info(
        f"Nightly batch completed for {label} in "
        f"{result.elapsed_seconds or 0.0:.1f}s: "
        f"dream_writes={dream.consolidated_count if dream else 0} "
        f"dream_proposals={dream.proposal_count if dream else 0} "
        f"ratified={ratification.ratified_count if ratification else 0} "
        f"superseded={ratification.superseded_count if ratification else 0}"
    )


def log_community_outcome(scope: MemoryScope, result: dict[str, Any]) -> None:
    if result.get("error"):
        logger.warning(
            f"Community rebuild errored for {scope_label(scope)}: {result['error']}"
        )
        return
    logger.info(
        f"Community rebuild completed for {scope_label(scope)} in "
        f"{result.get('elapsed_seconds') or 0.0:.1f}s: "
        f"{result.get('communities_built')}"
    )


def memory_job_kwargs(scope: MemoryScope) -> dict[str, str]:
    """Job kwargs of a dream-system cron. The account's keep the
    ``{"user_id": ...}`` they had when the crons were per user, so an
    existing job, a re-registered one and a rolled-back scheduler all call
    the body the same way."""
    if scope.expert_id is None:
        return {"user_id": scope.owner_user_id}
    return {"user_id": scope.owner_user_id, "expert_id": scope.expert_id}


def memory_job_result(
    scope: MemoryScope,
    job_id: str,
    next_run_time: datetime | None,
    user_timezone: str,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "user_id": scope.owner_user_id,
        "scope_key": scope.scope_key,
        "user_timezone": user_timezone,
        "next_run_time": next_run_time.isoformat() if next_run_time else None,
    }


def memory_job_skipped(
    scope: MemoryScope, user_timezone: str, reason: str
) -> dict[str, Any]:
    return {
        "id": None,
        "user_id": scope.owner_user_id,
        "scope_key": scope.scope_key,
        "user_timezone": user_timezone,
        "next_run_time": None,
        "skipped": True,
        "reason": reason,
    }


async def _may_fire(scope: MemoryScope, expert_status: ExpertStatusLookup) -> bool:
    if not await _schedule_active(scope):
        return False
    if scope.expert_id is None:
        return True
    status = await expert_status(scope.owner_user_id, scope.expert_id)
    if status != "active":
        logger.info(f"Memory cron of {scope_label(scope)} skipped: expert is {status}")
    return status == "active"


async def _schedule_active(scope: MemoryScope) -> bool:
    row = await memory_schedule_db().get_scope_schedule(
        scope.owner_user_id, scope.scope_key
    )
    if row is not None:
        return row.state == MemoryScopeScheduleState.ACTIVE
    if not scope.is_expert:
        return True
    logger.warning(f"No registry row for {scope_label(scope)}; removing its crons")
    await remove_scope_jobs(scope)
    return False
