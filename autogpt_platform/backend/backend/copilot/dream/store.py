"""The durable record of every dream pass, written as the pass moves.

Both routes write here, through ``db_accessors.dream_db()``: the Prisma
module where it is connected, the DatabaseManager RPC in the scheduler and
the batch executor. The sync orchestrator records the start, the gathered
window, each phase's output, the operations it is about to apply and how the
pass ended. The batch path records the submit, each phase that lands, each
next batch, the apply and the end, with what the landed phases used.

The row is additive: the lock, the batch state and the admin job status stay
in Redis as before. So a write that fails is logged at warning and dropped,
never raised; a pass must not fail because its record did.

``read_dream_pass`` and ``dream_pass_result_from_row`` are the read side, for
the admin API and the eval driver: a pass's ``DreamPassResult`` rebuilt from
its row, with usage and timings for batch passes too.
"""

import logging
from collections.abc import Callable
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
from backend.data.db_accessors import dream_db
from backend.data.dream_pass import (
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

from .fetch import DreamInput
from .routing import ExecutionPath
from .schemas import DreamOperations, DreamPassResult, DreamPassUsage, DreamPhase

logger = logging.getLogger(__name__)

# What started a pass: the nightly job, the admin trigger, or an eval run.
DreamTrigger = Literal["cron", "admin", "eval"]

# Errors are capped as the admin job row caps them (job_status.mark_errored).
_MAX_ERROR_CHARS = 2000

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
# The step a pass is at once a phase's output is in.
_STEP_AFTER: dict[DreamPhase, DreamPassPhase] = {
    "consolidate": DreamPassPhase.RECOMBINE,
    "recombine": DreamPassPhase.SANITIZE,
    "sanitize": DreamPassPhase.APPLY,
}


async def start_pass(
    pass_id: str,
    scope: MemoryScope,
    *,
    route: ExecutionPath,
    trigger: DreamTrigger,
    started_at: datetime,
) -> None:
    """Insert the pass's row: running, gathering its input."""
    try:
        await dream_db().create_dream_pass(
            DreamPassDraft(
                id=pass_id,
                user_id=scope.owner_user_id,
                expert_id=scope.expert_id,
                scope_key=scope.scope_key,
                route=_ROUTES[route],
                trigger=_TRIGGERS[trigger],
                started_at=started_at,
            )
        )
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not insert its record; the pass goes on",
            exc_info=True,
        )


async def record_gathered(pass_id: str, input_bundle: DreamInput) -> None:
    """The input is in: the window it covers; consolidate is next."""
    await _write(
        pass_id,
        "the gathered window",
        lambda: DreamPassUpdate(
            phase=DreamPassPhase.CONSOLIDATE,
            window_start=input_bundle.window_start,
            window_end=input_bundle.window_end,
        ),
    )


async def record_phase_output(
    pass_id: str, phase: DreamPhase, output: BaseModel
) -> None:
    """A phase's validated output, on either route; the pass moves on."""
    await _write(
        pass_id,
        f"the {phase} output",
        lambda: DreamPassUpdate(
            phase=_STEP_AFTER[phase],
            phase_outputs=DreamPhaseOutputs.model_validate({phase: output}),
        ),
    )


async def record_applying(pass_id: str, ops: DreamOperations) -> None:
    """Apply is about to run on *ops*, the clamped operations."""
    await _write(
        pass_id,
        "the start of apply",
        lambda: DreamPassUpdate(
            status=DreamPassStatus.APPLYING,
            phase=DreamPassPhase.APPLY,
            operations=DreamPassOperations(planned=ops),
        ),
    )


async def record_submitted(
    pass_id: str,
    *,
    input_bundle: DreamInput,
    provider_batch_id: str,
    lease_token: str,
    lease_ttl_seconds: int,
) -> None:
    """The batch route took the pass: consolidate's batch is in flight, and
    the dream lock is held for the callbacks until it lapses."""
    now = datetime.now(timezone.utc)
    await _write(
        pass_id,
        "the batch submit",
        lambda: DreamPassUpdate(
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.CONSOLIDATE,
            provider_batch_id=provider_batch_id,
            input_bundle=input_bundle,
            lease_token=lease_token,
            lease_expires_at=now + timedelta(seconds=lease_ttl_seconds),
            submitted_at=now,
        ),
    )


async def record_next_batch(pass_id: str, provider_batch_id: str) -> None:
    """The next phase's batch is in flight."""
    await _write(
        pass_id,
        "the next batch",
        lambda: DreamPassUpdate(provider_batch_id=provider_batch_id),
    )


async def record_sync_outcome(result: DreamPassResult) -> None:
    """How a pass the orchestrator ran ended. A pass it handed to the batch
    route stays open: the batch callbacks write its end."""
    handed_to_batch = (
        result.execution_path == "anthropic_batch"
        and not result.skipped
        and result.error is None
    )
    if handed_to_batch:
        return
    await _write(result.pass_id, "the outcome", lambda: _outcome(result, result.usage))


async def record_batch_complete(
    result: DreamPassResult, usage: DreamPassUsage | None
) -> None:
    """A batch pass applied: *result* holds what apply reported, *usage* what
    its phases used."""
    await _write(result.pass_id, "the outcome", lambda: _outcome(result, usage))


async def record_batch_failed(
    pass_id: str, error: str, usage: DreamPassUsage | None
) -> None:
    """A batch pass failed; *usage* is what its landed phases used."""
    await _write(
        pass_id,
        "the failure",
        lambda: _failed(error, usage, datetime.now(timezone.utc)),
    )


async def read_dream_pass(pass_id: str, *, user_id: str) -> DreamPassRecord | None:
    """The pass's row when *user_id* owns it, else ``None``. Unlike the
    writes, a failed read raises: its caller is answering a request."""
    return await dream_db().get_dream_pass_for_user(pass_id, user_id)


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


async def _write(pass_id: str, step: str, build: Callable[[], DreamPassUpdate]) -> None:
    """Write one transition, building it inside the guard so no part of it
    can fail the pass."""
    try:
        written = await dream_db().update_dream_pass(pass_id, build())
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not record {step}; the pass goes on",
            exc_info=True,
        )
        return
    if not written:
        logger.warning(f"Dream pass {pass_id}: no open record to write {step} to")


def _outcome(result: DreamPassResult, usage: DreamPassUsage | None) -> DreamPassUpdate:
    finished = result.completed_at or datetime.now(timezone.utc)
    if result.skipped:
        return DreamPassUpdate(
            status=DreamPassStatus.SKIPPED,
            skip_reason=result.skip_reason,
            completed_at=finished,
        )
    if result.error is not None:
        return _failed(result.error, usage, finished)
    return DreamPassUpdate(
        status=DreamPassStatus.COMPLETE,
        phase=DreamPassPhase.DONE,
        operations=DreamPassOperations(applied=_applied(result)),
        usage=usage,
        applied_at=finished,
        completed_at=finished,
    )


def _failed(
    error: str, usage: DreamPassUsage | None, finished: datetime
) -> DreamPassUpdate:
    return DreamPassUpdate(
        status=DreamPassStatus.ERRORED,
        error=error[:_MAX_ERROR_CHARS],
        usage=usage,
        completed_at=finished,
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
