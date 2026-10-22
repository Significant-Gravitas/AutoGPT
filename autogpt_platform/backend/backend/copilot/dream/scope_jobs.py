"""The dream-system crons of one memory scope, against the scheduler and the
``MemoryScopeSchedule`` row: registering and recording them, and removing
them.

``registry.py`` decides when a scope's crons should exist; this module does
the work, and ``scope_crons.py`` holds what the cron bodies ask of it. Every
call to a dependency runs under the registry deadline (``deadline.py``).
Everything fails soft: logged, never raised.

Turning a flag off leaves the scope's recorded crons in place. Their bodies
check the flag at run time and do nothing. Pause or wipe removes them.
Cleaning them up on flag-off needs generation-aware intent, which is left for
later. (A pass that re-registers, on a timezone change or a forced refresh,
stops recording a cron whose flag is off and leaves the cron itself, unless
the reconcile had registered it, when its next pass takes it back.)
"""

import logging
from typing import Any, Iterable

from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.util.clients import get_scheduler_client
from backend.util.feature_flag import evaluate_feature_flag

from .deadline import within_deadline
from .scheduling import DREAM_SYSTEM_JOBS, DreamSystemJob, jobs_for

logger = logging.getLogger(__name__)


async def flag_gate(scope: MemoryScope) -> tuple[dict[str, Any], list[DreamSystemJob]]:
    """Split the scope's crons (``jobs_for``) on the owner's flags: skip
    results for the disabled ones, and the enabled ones still to consider."""
    results: dict[str, Any] = {}
    enabled: list[DreamSystemJob] = []
    for job in jobs_for(scope):
        if job.flag is None:
            enabled.append(job)
            continue
        try:
            on, authoritative = await within_deadline(
                evaluate_feature_flag(job.flag, scope.owner_user_id)
            )
        except Exception:
            logger.warning(f"Flag check failed for {job.name}", exc_info=True)
            results[job.job_id_prefix] = skipped("flag_unavailable")
            continue
        if not authoritative:
            results[job.job_id_prefix] = skipped("flag_unavailable")
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
    skipped_jobs: dict[str, Any],
) -> dict[str, Any]:
    """Claim the scope's row if it has none, register every enabled cron
    that is missing or was registered in another timezone, and record the
    job ids on the row. One result per enabled cron, keyed by prefix."""
    if row is None:
        row = await _claim(scope, user_timezone)
        if row is None:
            return skip_all(enabled, "registry_unavailable")
        if row.state != MemoryScopeScheduleState.ACTIVE:
            return skip_all(enabled, state_reason(row))
    stale = force_refresh or row.timezone != user_timezone
    if row.timezone != user_timezone:
        failure = await _record_timezone(scope, row, user_timezone)
        if failure is not None:
            return skip_all(enabled, failure)
    outcomes: dict[str, Any] = dict(skipped_jobs)
    for job in enabled:
        if not stale and recorded_job_id(row, job) is not None:
            outcomes[job.job_id_prefix] = None
            continue
        outcomes[job.job_id_prefix] = await _register_job(scope, job, user_timezone)
    return await _record(scope, row, user_timezone, outcomes, stale)


async def _record_timezone(
    scope: MemoryScope, row: MemoryScopeSchedule, user_timezone: str
) -> str | None:
    """Persist the new timezone before replacing crons, so reconciliation
    retains it even if the final job-id write fails."""
    try:
        recorded = await within_deadline(
            memory_schedule_db().record_scope_jobs(
                scope.owner_user_id,
                scope.scope_key,
                user_timezone=user_timezone,
                community_job_id=row.community_job_id,
                nightly_job_id=row.nightly_job_id,
                learning_job_id=row.learning_job_id,
            )
        )
    except Exception:
        logger.warning(
            f"Could not record timezone of {scope_label(scope)}", exc_info=True
        )
        return "record_failed"
    return None if recorded else "scope_changed"


async def remove_scope_jobs(scope: MemoryScope) -> bool:
    """Remove every dream-system cron of the scope from the scheduler;
    whether the scheduler confirmed it."""
    try:
        await within_deadline(
            get_scheduler_client().remove_scope_memory_jobs(scope=scope)
        )
    except Exception:
        logger.warning(
            f"Could not remove the crons of {scope_label(scope)}", exc_info=True
        )
        return False
    return True


def skipped(reason: str) -> dict[str, Any]:
    return {"skipped": True, "reason": reason}


def skip_all(jobs: Iterable[DreamSystemJob], reason: str) -> dict[str, Any]:
    return {job.job_id_prefix: skipped(reason) for job in jobs}


def state_reason(row: MemoryScopeSchedule) -> str:
    return f"scope_{row.state.value.lower()}"


def scope_label(scope: MemoryScope) -> str:
    owner = f"user {scope.owner_user_id[:12]}"
    return f"{owner} expert {scope.expert_id[:12]}" if scope.expert_id else owner


async def _claim(scope: MemoryScope, user_timezone: str) -> MemoryScopeSchedule | None:
    try:
        return await within_deadline(
            memory_schedule_db().claim_scope_schedule(
                scope.owner_user_id, scope.scope_key, scope.expert_id, user_timezone
            )
        )
    except Exception:
        logger.warning(f"Could not claim {scope_label(scope)}", exc_info=True)
        return None


async def _register_job(
    scope: MemoryScope, job: DreamSystemJob, user_timezone: str
) -> dict[str, Any]:
    try:
        result = await within_deadline(
            job.register(get_scheduler_client(), scope, user_timezone)
        )
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
) -> dict[str, Any]:
    """Write the job ids this pass settled on and return the outcomes as
    they stand afterwards: a registration the row could not record is a
    failure, and one the scope left ACTIVE under is ``scope_changed``. Its
    crons are left where they are: the fire gate holds a scope that is not
    ACTIVE, the reconcile removes them (``scope_reconcile.py``), and taking
    them down here could take a concurrent resume's crons with them."""
    ids = {
        job.row_field: _next_job_id(row, job, outcomes.get(job.job_id_prefix), stale)
        for job in DREAM_SYSTEM_JOBS
    }
    if not stale and all(
        ids[job.row_field] == recorded_job_id(row, job) for job in DREAM_SYSTEM_JOBS
    ):
        return outcomes
    try:
        recorded = await within_deadline(
            memory_schedule_db().record_scope_jobs(
                scope.owner_user_id,
                scope.scope_key,
                user_timezone=user_timezone,
                community_job_id=ids["community_job_id"],
                nightly_job_id=ids["nightly_job_id"],
                learning_job_id=ids["learning_job_id"],
            )
        )
    except Exception:
        logger.warning(
            f"Could not record the jobs of {scope_label(scope)}", exc_info=True
        )
        return _replace_registered(outcomes, "record_failed")
    if recorded:
        return outcomes
    logger.info(
        f"{scope_label(scope)} left ACTIVE while registering; the reconcile "
        "removes its crons"
    )
    return _replace_registered(outcomes, "scope_changed")


def _replace_registered(outcomes: dict[str, Any], reason: str) -> dict[str, Any]:
    """The outcomes with every registration made this pass turned into a
    skip with ``reason``."""
    return {
        prefix: skipped(reason) if _registered(outcome) else outcome
        for prefix, outcome in outcomes.items()
    }


def _registered(outcome: Any) -> bool:
    return bool(outcome) and not outcome.get("skipped") and bool(outcome.get("id"))


def _next_job_id(
    row: MemoryScopeSchedule, job: DreamSystemJob, outcome: Any, stale: bool
) -> str | None:
    """Keep failed registration intent so reconciliation can retry it."""
    if _registered(outcome):
        return outcome["id"]
    if outcome and outcome.get("reason") in (
        "registration_failed",
        "flag_unavailable",
    ):
        return recorded_job_id(row, job)
    return None if stale else recorded_job_id(row, job)


def recorded_job_id(row: MemoryScopeSchedule, job: DreamSystemJob) -> str | None:
    """The job id ``row`` records for ``job``, None when it records none."""
    match job.row_field:
        case "community_job_id":
            return row.community_job_id
        case "nightly_job_id":
            return row.nightly_job_id
        case "learning_job_id":
            return row.learning_job_id
