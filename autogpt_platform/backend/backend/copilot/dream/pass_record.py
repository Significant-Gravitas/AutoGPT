"""What each step of a dream pass writes to its ``DreamPass`` row, and what
the row reads back as.

Pure: how a route, a trigger and a step are named on the row, the insert or
update each transition makes (``store.py`` writes them, bounded), including
the lease renewal and the two that stop a pass from outside (a cancel, an
expiry), and what a row reads back as: the ``DreamPassResult`` it describes,
for the admin API and the eval driver, and whether it says its pass was
stopped. Every transition that closes a row drops its lease and its input
bundle (``CLOSED_ROW_CLEARS``).
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import (
    CLOSED_ROW_CLEARS,
    INITIAL_CANCEL_GENERATION,
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

from .fetch import DreamInput
from .locks import DEFAULT_LOCK_TTL_SECONDS
from .routing import ExecutionPath
from .schemas import DreamOperations, DreamPassResult, DreamPassUsage, DreamPhase

# What started a pass: the nightly job, the admin trigger, or an eval run.
DreamTrigger = Literal["cron", "admin", "eval"]

# Errors are capped as the admin job row caps them (job_status.mark_errored).
MAX_ERROR_CHARS = 2000

_ROUTES: dict[ExecutionPath, DreamPassRoute] = {
    "sync_baseline": DreamPassRoute.SYNC,
    "anthropic_batch": DreamPassRoute.ANTHROPIC_BATCH,
}
_EXECUTION_PATHS: dict[DreamPassRoute, ExecutionPath] = {
    route: path for path, route in _ROUTES.items()
}
_TRIGGERS: dict[DreamTrigger, DreamPassTrigger] = {
    "cron": DreamPassTrigger.CRON,
    "admin": DreamPassTrigger.ADMIN,
    "eval": DreamPassTrigger.EVAL,
}
# The step a pass is at while one of its phases runs, or waits on its batch.
_PHASE_STEPS: dict[DreamPhase, DreamPassPhase] = {
    "consolidate": DreamPassPhase.CONSOLIDATE,
    "recombine": DreamPassPhase.RECOMBINE,
    "sanitize": DreamPassPhase.SANITIZE,
}
# The step a pass is at once a phase's output is in.
_STEP_AFTER: dict[DreamPhase, DreamPassPhase] = {
    "consolidate": DreamPassPhase.RECOMBINE,
    "recombine": DreamPassPhase.SANITIZE,
    "sanitize": DreamPassPhase.APPLY,
}


def new_pass(
    pass_id: str,
    scope: MemoryScope,
    *,
    route: ExecutionPath,
    trigger: DreamTrigger,
    started_at: datetime,
    lease_token: str,
) -> DreamPassDraft:
    """A pass's first row: running, gathering its input, with the lease of
    the lock it is about to take under *lease_token* (the sync TTL from its
    start)."""
    return DreamPassDraft(
        id=pass_id,
        user_id=scope.owner_user_id,
        expert_id=scope.expert_id,
        scope_key=scope.scope_key,
        route=_ROUTES[route],
        trigger=_TRIGGERS[trigger],
        started_at=started_at,
        lease_token=lease_token,
        lease_expires_at=started_at + timedelta(seconds=DEFAULT_LOCK_TTL_SECONDS),
    )


def lease(lease_token: str, ttl_seconds: int) -> DreamPassUpdate:
    """The pass renewed its lock under *lease_token* for *ttl_seconds*."""
    return DreamPassUpdate(
        lease_token=lease_token,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
    )


def gathered(input_bundle: DreamInput) -> DreamPassUpdate:
    """The input is in: the window it covers; consolidate is next."""
    return DreamPassUpdate(
        phase=DreamPassPhase.CONSOLIDATE,
        window_start=input_bundle.window_start,
        window_end=input_bundle.window_end,
    )


def phase_output(phase: DreamPhase, output: BaseModel) -> DreamPassUpdate:
    """A phase's validated output, on either route; the pass moves on."""
    return DreamPassUpdate(
        phase=_STEP_AFTER[phase],
        phase_outputs=DreamPhaseOutputs.model_validate({phase: output}),
    )


def applying(ops: DreamOperations) -> DreamPassUpdate:
    """Apply is about to run on *ops*, the clamped operations."""
    return DreamPassUpdate(
        status=DreamPassStatus.APPLYING,
        phase=DreamPassPhase.APPLY,
        operations=DreamPassOperations(planned=ops),
    )


def submitted(
    *,
    input_bundle: DreamInput,
    provider_batch_id: str,
    lease_token: str,
    lease_ttl_seconds: int,
) -> DreamPassUpdate:
    """The batch route took the pass: consolidate's batch is in flight, and the
    dream lock is held for the callbacks until it lapses."""
    now = datetime.now(timezone.utc)
    return DreamPassUpdate(
        status=DreamPassStatus.SUBMITTED,
        phase=_PHASE_STEPS["consolidate"],
        provider_batch_id=provider_batch_id,
        input_bundle=input_bundle,
        lease_token=lease_token,
        lease_expires_at=now + timedelta(seconds=lease_ttl_seconds),
        submitted_at=now,
    )


def next_batch(phase: DreamPhase, provider_batch_id: str) -> DreamPassUpdate:
    """*phase*'s batch is in flight."""
    return DreamPassUpdate(
        phase=_PHASE_STEPS[phase], provider_batch_id=provider_batch_id
    )


def handed_to_batch(result: DreamPassResult) -> bool:
    """The sync entry point handed the pass to the batch route: its end is the
    batch callbacks' to write."""
    return (
        result.execution_path == "anthropic_batch"
        and not result.skipped
        and result.error is None
    )


def outcome(result: DreamPassResult, usage: DreamPassUsage | None) -> DreamPassUpdate:
    """How a pass ended: skipped, failed, or applied."""
    finished = result.completed_at or datetime.now(timezone.utc)
    if result.skipped:
        return DreamPassUpdate(
            status=DreamPassStatus.SKIPPED,
            skip_reason=result.skip_reason,
            completed_at=finished,
            clear=CLOSED_ROW_CLEARS,
        )
    if result.error is not None:
        return failed(result.error, usage, finished)
    return DreamPassUpdate(
        status=DreamPassStatus.COMPLETE,
        phase=DreamPassPhase.DONE,
        operations=DreamPassOperations(applied=_applied(result)),
        usage=usage,
        applied_at=finished,
        completed_at=finished,
        clear=CLOSED_ROW_CLEARS,
    )


def failed(
    error: str, usage: DreamPassUsage | None, finished: datetime
) -> DreamPassUpdate:
    return DreamPassUpdate(
        status=DreamPassStatus.ERRORED,
        error=error[:MAX_ERROR_CHARS],
        usage=usage,
        completed_at=finished,
        clear=CLOSED_ROW_CLEARS,
    )


def cancelled(reason: str, *, owner_user_id: str) -> DreamPassUpdate:
    """A cancel: the owner's open pass closes CANCELLED with *reason* as its
    error, its cancel generation bumped so the running pass stops at its next
    check. Written only while the row is open and the owner's."""
    return DreamPassUpdate(
        status=DreamPassStatus.CANCELLED,
        error=reason[:MAX_ERROR_CHARS],
        completed_at=datetime.now(timezone.utc),
        bump_cancel_generation=True,
        owner_user_id=owner_user_id,
        clear=CLOSED_ROW_CLEARS,
    )


def expired(reason: str, *, not_updated_since: datetime | None) -> DreamPassUpdate:
    """An open pass closed EXPIRED from outside (a newer pass's guard, the
    reaper, a duplicate delivery), its cancel generation bumped like a
    cancel's. Written only while the row is open and, given
    *not_updated_since* (the stale row as it was read), only if nothing has
    written it since; an admin's forced expiry gives ``None``."""
    return DreamPassUpdate(
        status=DreamPassStatus.EXPIRED,
        error=reason[:MAX_ERROR_CHARS],
        completed_at=datetime.now(timezone.utc),
        bump_cancel_generation=True,
        not_updated_since=not_updated_since,
        clear=CLOSED_ROW_CLEARS,
    )


def stop_error(row: DreamPassRecord) -> str | None:
    """Why the row says its pass was stopped from outside, as the pass's
    error (``cancelled: <reason>``, ``expired: <reason>``); ``None`` while its
    cancel generation is still the one a new row starts at."""
    if row.cancel_generation == INITIAL_CANCEL_GENERATION:
        return None
    how = row.status.value.lower()
    return f"{how}: {row.error}" if row.error else how


def dream_pass_result_from_row(row: DreamPassRecord) -> DreamPassResult:
    """The ``DreamPassResult`` a pass's row describes, on either route.

    A pass still in flight reads as neither skipped nor errored, with no
    completion time and no usage yet.
    """
    applied = row.operations.applied or DreamPassApplied()
    return DreamPassResult(
        user_id=row.user_id,
        pass_id=row.id,
        started_at=row.started_at,
        completed_at=row.completed_at,
        elapsed_seconds=_elapsed_seconds(row),
        execution_path=_EXECUTION_PATHS[row.route],
        consolidated_count=applied.consolidated_count,
        proposal_count=applied.proposal_count,
        demotion_count=applied.demotion_count,
        entity_invalidation_count=applied.entity_invalidation_count,
        summary_for_user=applied.summary_for_user,
        dream_session_id=applied.dream_session_id,
        ingestion_drain_status=applied.ingestion_drain_status,
        operations=applied.snapshot,
        usage=row.usage,
        error=row.error,
        skipped=row.status == DreamPassStatus.SKIPPED,
        skip_reason=row.skip_reason,
    )


def _applied(result: DreamPassResult) -> DreamPassApplied:
    return DreamPassApplied(
        consolidated_count=result.consolidated_count,
        proposal_count=result.proposal_count,
        demotion_count=result.demotion_count,
        entity_invalidation_count=result.entity_invalidation_count,
        summary_for_user=result.summary_for_user,
        dream_session_id=result.dream_session_id,
        ingestion_drain_status=result.ingestion_drain_status,
        snapshot=result.operations,
    )


def _elapsed_seconds(row: DreamPassRecord) -> float | None:
    if row.started_at is None or row.completed_at is None:
        return None
    return (row.completed_at - row.started_at).total_seconds()
