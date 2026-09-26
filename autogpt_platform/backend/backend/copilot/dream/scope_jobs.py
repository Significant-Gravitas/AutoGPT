"""The dream-system crons of one memory scope, against the scheduler and the
``MemoryScopeSchedule`` row: registering and recording them, removing them,
and the gate their cron bodies ask before running.

``registry.py`` decides when a scope's crons should exist; this module does
the work. The cron bodies in ``executor/scheduler.py`` call the last three
public functions. Everything fails soft: logged, never raised.
"""

import logging
from typing import Any, Iterable

from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule, ScopeRunKind
from backend.util.clients import get_scheduler_client
from backend.util.feature_flag import is_feature_enabled

from .scheduling import (
    DREAM_SYSTEM_JOBS,
    DreamSystemJob,
    clear_registration_marker,
    write_registration_marker,
)

logger = logging.getLogger(__name__)


async def flag_gate(scope: MemoryScope) -> tuple[dict[str, Any], list[DreamSystemJob]]:
    """Split the crons on the owner's flags: skip results for the disabled
    ones, and the enabled ones still to consider."""
    results: dict[str, Any] = {}
    enabled: list[DreamSystemJob] = []
    for job in DREAM_SYSTEM_JOBS:
        try:
            on = await is_feature_enabled(job.flag, scope.owner_user_id)
        except Exception:
            logger.warning(f"Flag check failed for {job.name}", exc_info=True)
            results[job.job_id_prefix] = skipped("registration_failed")
            continue
        if on:
            enabled.append(job)
        else:
            results[job.job_id_prefix] = skipped(job.skip_reason)
    return results, enabled


async def register_scope_jobs(
    scope: MemoryScope,
    enabled: list[DreamSystemJob],
    row: MemoryScopeSchedule | None,
    user_timezone: str,
    force_refresh: bool,
) -> dict[str, Any]:
    """Claim the scope's row if it has none, register every enabled cron
    that is missing or was registered in another timezone, and record the
    job ids on the row. One result per enabled cron, keyed by prefix."""
    if row is None:
        try:
            row = await memory_schedule_db().claim_scope_schedule(
                scope.owner_user_id, scope.scope_key, scope.expert_id, user_timezone
            )
        except Exception:
            logger.warning(f"Could not claim {scope_label(scope)}", exc_info=True)
            return skip_all(enabled, "registry_unavailable")
        if row.state != MemoryScopeScheduleState.ACTIVE:
            return skip_all(enabled, state_reason(row))
    stale = force_refresh or row.timezone != user_timezone
    outcomes: dict[str, Any] = {}
    for job in enabled:
        if not stale and _recorded_job_id(row, job) is not None:
            outcomes[job.job_id_prefix] = None
            continue
        outcomes[job.job_id_prefix] = await _register_job(scope, job, user_timezone)
    await _record(scope, row, user_timezone, outcomes, stale)
    return outcomes


async def remove_scope_jobs(scope: MemoryScope) -> None:
    """Remove every dream-system cron of the scope from the scheduler."""
    try:
        await get_scheduler_client().remove_scope_memory_jobs(scope=scope)
    except Exception:
        logger.warning(
            f"Could not remove the crons of {scope_label(scope)}", exc_info=True
        )


async def scope_schedule_active(scope: MemoryScope) -> bool:
    """Whether the registry lets the scope's crons fire now.

    An ACTIVE row says yes; PAUSED and WIPED say no. No row on the account
    means crons from before the registry, which run as they always did. No
    row on an expert means the expert or its owner was deleted (the row
    cascades), so its orphaned crons are removed. A failed read says no.
    """
    try:
        row = await memory_schedule_db().get_scope_schedule(
            scope.owner_user_id, scope.scope_key
        )
    except Exception:
        logger.warning(f"Registry read failed for {scope_label(scope)}", exc_info=True)
        return False
    if row is not None:
        return row.state == MemoryScopeScheduleState.ACTIVE
    if not scope.is_expert:
        return True
    logger.warning(f"No registry row for {scope_label(scope)}; removing its crons")
    await remove_scope_jobs(scope)
    return False


async def record_scope_run(scope: MemoryScope, kind: ScopeRunKind) -> None:
    """Stamp the time a cron body ran its pass without an error."""
    try:
        await memory_schedule_db().record_scope_run(
            scope.owner_user_id, scope.scope_key, kind
        )
    except Exception:
        logger.warning(
            f"Could not stamp the {kind} run of {scope_label(scope)}", exc_info=True
        )


async def forget_registration(scope: MemoryScope, job: DreamSystemJob) -> None:
    """After ``job`` was deleted outside the registry, drop its id from the
    row and its Redis marker so the next ensure registers it again."""
    await clear_registration_marker(scope, job.registration_key_prefix)
    try:
        await memory_schedule_db().forget_scope_job(
            scope.owner_user_id, scope.scope_key, job.job_id(scope)
        )
    except Exception:
        logger.warning(
            f"Could not forget {job.name} of {scope_label(scope)}", exc_info=True
        )


def skipped(reason: str) -> dict[str, Any]:
    return {"skipped": True, "reason": reason}


def skip_all(jobs: Iterable[DreamSystemJob], reason: str) -> dict[str, Any]:
    return {job.job_id_prefix: skipped(reason) for job in jobs}


def state_reason(row: MemoryScopeSchedule) -> str:
    return f"scope_{row.state.value.lower()}"


def scope_label(scope: MemoryScope) -> str:
    owner = f"user {scope.owner_user_id[:12]}"
    return f"{owner} expert {scope.expert_id[:12]}" if scope.expert_id else owner


async def _register_job(
    scope: MemoryScope, job: DreamSystemJob, user_timezone: str
) -> dict[str, Any]:
    try:
        result = await job.register(get_scheduler_client(), scope, user_timezone)
    except Exception:
        logger.warning(
            f"Dream-system: failed to register {job.name} for {scope_label(scope)}",
            exc_info=True,
        )
        return skipped("registration_failed")
    if result.get("skipped"):
        # The scheduler's own flag gate refused although ours said on (LD
        # cold start, targeting divergence): record nothing, retry next time.
        logger.warning(
            f"Dream-system: scheduler skipped {job.name} for {scope_label(scope)} "
            f"(reason={result.get('reason')}) despite the local flag check"
        )
        return result
    await write_registration_marker(scope, job.registration_key_prefix, user_timezone)
    logger.info(
        f"Dream-system: registered {job.name} for {scope_label(scope)} "
        f"(tz={user_timezone})"
    )
    return result


async def _record(
    scope: MemoryScope,
    row: MemoryScopeSchedule,
    user_timezone: str,
    outcomes: dict[str, Any],
    stale: bool,
) -> None:
    """Write the job ids this pass settled on; take the new jobs down again
    if the scope stopped being ACTIVE meanwhile."""
    ids = {
        job.row_field: _next_job_id(row, job, outcomes.get(job.job_id_prefix), stale)
        for job in DREAM_SYSTEM_JOBS
    }
    if not stale and all(
        ids[job.row_field] == _recorded_job_id(row, job) for job in DREAM_SYSTEM_JOBS
    ):
        return
    try:
        recorded = await memory_schedule_db().record_scope_jobs(
            scope.owner_user_id,
            scope.scope_key,
            user_timezone=user_timezone,
            community_job_id=ids["community_job_id"],
            nightly_job_id=ids["nightly_job_id"],
        )
    except Exception:
        logger.warning(
            f"Could not record the jobs of {scope_label(scope)}", exc_info=True
        )
        return
    if not recorded:
        logger.info(f"{scope_label(scope)} left ACTIVE while registering; removing")
        await remove_scope_jobs(scope)


def _next_job_id(
    row: MemoryScopeSchedule, job: DreamSystemJob, outcome: Any, stale: bool
) -> str | None:
    """A new registration's id; otherwise the recorded id, unless that job
    was registered in a timezone that no longer holds."""
    if outcome and not outcome.get("skipped") and outcome.get("id"):
        return outcome["id"]
    return None if stale else _recorded_job_id(row, job)


def _recorded_job_id(row: MemoryScopeSchedule, job: DreamSystemJob) -> str | None:
    if job.row_field == "community_job_id":
        return row.community_job_id
    return row.nightly_job_id
