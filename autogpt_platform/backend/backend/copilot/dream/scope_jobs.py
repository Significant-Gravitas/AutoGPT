"""The dream-system crons of one memory scope, against the scheduler and the
``MemoryScopeSchedule`` row: registering and recording them, and removing
them.

``registry.py`` decides when a scope's crons should exist; this module does
the work, and ``scope_crons.py`` holds what the cron bodies ask of it. Every
call to a dependency runs under the registry deadline (``deadline.py``).
Everything fails soft: logged, never raised.

Each cron's flag has one of three answers (``flag_gate``): on, and the cron
is registered; authoritatively off, and the cron the row records is
removed, its record first; unknown (the lookup failed, or answered only
with its default), and the record and the job stay as they are.
"""

import logging
from typing import Any, Iterable

from prisma.enums import MemoryScopeScheduleState
from pydantic import BaseModel, Field

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.util.clients import get_scheduler_client
from backend.util.feature_flag import evaluate_feature_flag

from .deadline import within_deadline
from .scheduling import DREAM_SYSTEM_JOBS, DreamSystemJob, jobs_for

logger = logging.getLogger(__name__)


class FlagGate(BaseModel):
    """The owner's flags over a scope's crons (``jobs_for``): ``enabled``
    are to be registered, ``disabled`` are authoritatively off, so the cron
    the row records goes, and ``skips`` holds a result for each cron not
    enabled. A cron whose flag could not be read is in neither list."""

    enabled: list[DreamSystemJob] = Field(default_factory=list)
    disabled: list[DreamSystemJob] = Field(default_factory=list)
    skips: dict[str, Any] = Field(default_factory=dict)


async def flag_gate(scope: MemoryScope) -> FlagGate:
    """Read the owner's flag of each of the scope's crons, each under the
    registry deadline."""
    gate = FlagGate()
    for job in jobs_for(scope):
        try:
            on, authoritative = await within_deadline(
                evaluate_feature_flag(job.flag, scope.owner_user_id)
            )
        except Exception:
            logger.warning(f"Flag check failed for {job.name}", exc_info=True)
            on, authoritative = False, False
        if on:
            gate.enabled.append(job)
        elif authoritative:
            gate.disabled.append(job)
            gate.skips[job.job_id_prefix] = skipped(job.skip_reason)
        else:
            gate.skips[job.job_id_prefix] = skipped("flag_unavailable")
    return gate


async def register_scope_jobs(
    scope: MemoryScope,
    gate: FlagGate,
    row: MemoryScopeSchedule | None,
    user_timezone: str,
    force_refresh: bool,
) -> dict[str, Any]:
    """Claim the scope's row if it has none, register every enabled cron
    that is missing or was registered in another timezone, remove every
    recorded cron whose flag is off, and record the job ids on the row. One
    result per enabled cron, and per off one whose removal failed, keyed by
    prefix."""
    if row is None:
        row = await _claim(scope, user_timezone)
        if row is None:
            return skip_all(gate.enabled, "registry_unavailable")
        if row.state != MemoryScopeScheduleState.ACTIVE:
            return skip_all(gate.enabled, state_reason(row))
    stale = force_refresh or row.timezone != user_timezone
    outcomes: dict[str, Any] = {}
    for job in gate.enabled:
        if not stale and recorded_job_id(row, job) is not None:
            outcomes[job.job_id_prefix] = None
            continue
        outcomes[job.job_id_prefix] = await _register_job(scope, job, user_timezone)
    removals = await _remove_disabled(scope, row, gate.disabled, outcomes)
    return await _record(scope, row, user_timezone, outcomes, removals, stale)


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


class _Removals(BaseModel):
    """The recorded crons whose flag is off that this pass tried to remove,
    by prefix, and those of them now gone."""

    tried: set[str] = Field(default_factory=set)
    gone: set[str] = Field(default_factory=set)


async def _remove_disabled(
    scope: MemoryScope,
    row: MemoryScopeSchedule,
    disabled: list[DreamSystemJob],
    outcomes: dict[str, Any],
) -> _Removals:
    """Remove each cron the row records whose flag is authoritatively off,
    its record first: the reconcile restores a cron a row records and takes
    back only one it registered itself, so the record goes before the job.
    A cron whose removal is not confirmed keeps its record (``_record``
    writes it back), so the next ensure tries again; it reports
    ``removal_failed``."""
    removals = _Removals()
    for job in disabled:
        if recorded_job_id(row, job) is None:
            continue
        removals.tried.add(job.job_id_prefix)
        if await _remove_job(scope, job):
            removals.gone.add(job.job_id_prefix)
        else:
            outcomes[job.job_id_prefix] = skipped("removal_failed")
    return removals


async def _remove_job(scope: MemoryScope, job: DreamSystemJob) -> bool:
    try:
        await within_deadline(
            memory_schedule_db().forget_scope_job(
                scope.owner_user_id, scope.scope_key, job.job_id(scope)
            )
        )
        await within_deadline(
            get_scheduler_client().remove_scope_memory_job(
                scope=scope, job_id_prefix=job.job_id_prefix
            )
        )
    except Exception:
        logger.warning(
            f"Could not remove {job.name} of {scope_label(scope)}", exc_info=True
        )
        return False
    logger.info(f"Dream-system: removed {job.name} of {scope_label(scope)}: flag off")
    return True


async def _record(
    scope: MemoryScope,
    row: MemoryScopeSchedule,
    user_timezone: str,
    outcomes: dict[str, Any],
    removals: _Removals,
    stale: bool,
) -> dict[str, Any]:
    """Write the job ids this pass settled on and return the outcomes as
    they stand afterwards: a registration the row could not record is a
    failure, and one the scope left ACTIVE under is ``scope_changed``. Its
    crons are left where they are: the fire gate holds a scope that is not
    ACTIVE, the reconcile removes them (``scope_reconcile.py``), and taking
    them down here could take a concurrent resume's crons with them. A
    removal that was not confirmed may have forgotten its cron already, so
    its record is written back even when nothing else changed."""
    enabled = {prefix for prefix in outcomes if prefix not in removals.tried}
    ids = {
        job.row_field: _next_job_id(row, job, outcomes, enabled, removals, stale)
        for job in DREAM_SYSTEM_JOBS
    }
    unconfirmed = removals.tried - removals.gone
    if (
        not stale
        and not unconfirmed
        and all(
            ids[job.row_field] == recorded_job_id(row, job) for job in DREAM_SYSTEM_JOBS
        )
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
    row: MemoryScopeSchedule,
    job: DreamSystemJob,
    outcomes: dict[str, Any],
    enabled: set[str],
    removals: _Removals,
    stale: bool,
) -> str | None:
    """A new registration's id; None for a cron removed because its flag is
    off, and for an enabled one that was registered in a timezone that no
    longer holds and could not be registered again (the next ensure does);
    otherwise the recorded id: a cron whose flag could not be read, or whose
    removal was not confirmed, keeps it."""
    outcome: Any = outcomes.get(job.job_id_prefix)
    if _registered(outcome):
        return outcome["id"]
    if job.job_id_prefix in removals.gone:
        return None
    if stale and job.job_id_prefix in enabled:
        return None
    return recorded_job_id(row, job)


def recorded_job_id(row: MemoryScopeSchedule, job: DreamSystemJob) -> str | None:
    """The job id ``row`` records for ``job``, None when it records none."""
    match job.row_field:
        case "community_job_id":
            return row.community_job_id
        case "nightly_job_id":
            return row.nightly_job_id
        case "learning_job_id":
            return row.learning_job_id
