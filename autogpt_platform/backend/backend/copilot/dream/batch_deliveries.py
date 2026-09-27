"""The dream batch deliveries that never run the phase chain.

The BatchExecutor hands each finished dream batch to ``batch_callbacks``'s
phase handler, which lands the phase and chains the next one or applies.
Three kinds of delivery end the pass instead, each the way ``fail_pass`` ends
one: the admin job closed, the landed phases charged once (through the cost
gate), the lock released by compare-and-delete, the batch state and bundle
deleted.

  * A batch whose pass has already ended: cancelled or expired while the
    batch was in flight, or closed any other way. Before the executor polls
    it (``should_dispatch``) it reads the pass's row, bounded; a closed row
    means nobody waits for the batch, so it is cancelled at the provider and
    the executor drops it without polling or dispatching it: nothing chained,
    nothing applied, nothing charged beyond the phases that landed. An open
    row, a missing one or one the store cannot read in time dispatches as
    before, and the phase handler keeps its own stop checks.
  * A payload no phase handler can take (``end_dead_end``).
  * A repeat of a delivery whose apply already claimed the gate
    (``finish_duplicate``): apply is never run twice, and the row, which a
    first delivery that died after its claim left APPLYING, is closed
    EXPIRED rather than left for the reaper.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.data.dream_pass_models import OPEN_STATUSES, DreamPassRecord
from backend.executor.batch_executor import PendingEntry

from . import job_status
from .batch_costs import log_all_phase_costs, recorded_usage
from .batch_outcome import BatchPass, fail_pass, mark_job_errored, release_lock
from .batch_state import best_effort_cleanup
from .pass_record import stop_error
from .provider_batch import cancel_provider_batch
from .schemas import DreamOperations, DreamPassResult, IngestionDrainStatus
from .store import read_pass, record_batch_failed, record_expired

logger = logging.getLogger(__name__)

# The check's row read, shorter than a record write's deadline: the executor
# walks its queue serially and asks at every poll.
DISPATCH_CHECK_READ_TIMEOUT_SECONDS = 3.0

# How a pass whose last batch came back twice closes: the first delivery
# claimed apply and never finished.
DUPLICATE_ERROR = (
    "apply: an earlier delivery claimed apply and never finished; closed "
    "without applying again, and what it wrote may have landed"
)


async def should_dispatch(entry: PendingEntry) -> bool:
    """The executor's check before it polls one of a dream pass's batches:
    ``False`` once the pass's row has closed, after the batch is cancelled
    and the pass ended out (see the module docstring); ``True`` otherwise."""
    bp = BatchPass.from_payload(entry.payload or {})
    if not bp.pass_id:
        return True
    row = await _row(bp.pass_id)
    if row is None or row.status in OPEN_STATUSES:
        return True
    await _drop_closed(bp, row, entry.provider_batch_id)
    return False


async def end_dead_end(bp: BatchPass, error: str) -> None:
    """A payload no phase handler can take ends its pass as far as the
    payload names it: with an owner and a pass, like any failure. Without an
    owner there is nobody to charge, so the admin job and the record close
    and the pass's state goes; without a pass there is no token, so the lock
    is left to its TTL (``release_lock`` says so)."""
    if bp.user_id and bp.pass_id:
        await fail_pass(bp, error)
        return
    await mark_job_errored(bp.job_id, error, dead_end=True)
    if bp.pass_id:
        usage = await recorded_usage(bp.pass_id, bp.phase_models)
        await record_batch_failed(bp.pass_id, error, usage)
        await best_effort_cleanup(bp.pass_id)
    if bp.user_id:
        await release_lock(bp)


async def finish_duplicate(
    bp: BatchPass, state: dict[str, dict[str, Any]], ops: DreamOperations
) -> None:
    """A repeat of a delivery that already claimed the pass's apply: apply is
    skipped and the first delivery's results kept. That delivery normally
    closed the job and the row; one that died after its claim left both
    open, so the job is finalized with the attempted counts and the row
    closed EXPIRED (``DUPLICATE_ERROR``). Then as any end: the landed phases
    charged once, the lock released, the state and bundle deleted."""
    logger.info(
        "Duplicate dispatch for pass=%s — operations already applied; "
        "preserving the first delivery's job result",
        bp.pass_id,
    )
    await _finalize_stuck_duplicate(bp, ops)
    await record_expired(bp.pass_id, DUPLICATE_ERROR)
    await log_all_phase_costs(
        user_id=bp.user_id,
        expert_id=bp.expert_id,
        pass_id=bp.pass_id,
        state=state,
        phase_models=bp.phase_models,
    )
    await release_lock(bp)
    await best_effort_cleanup(bp.pass_id)


async def _drop_closed(
    bp: BatchPass, row: DreamPassRecord, provider_batch_id: str
) -> None:
    """End out a pass whose row closed while *provider_batch_id* was in
    flight, as its owner on the row. The row refuses the failure
    ``fail_pass`` writes: it keeps how it closed."""
    owner = bp.model_copy(update={"user_id": row.user_id, "expert_id": row.expert_id})
    reason = stop_error(row) or f"pass already {row.status.value.lower()}"
    logger.info(
        f"Dream batch {provider_batch_id} of pass {row.id} dropped unpolled: {reason}"
    )
    await cancel_provider_batch(provider_batch_id)
    await fail_pass(owner, reason)


async def _row(pass_id: str) -> DreamPassRecord | None:
    """The pass's row, or ``None`` when there is none or the store cannot say
    in time, and the batch is dispatched as before."""
    try:
        return await read_pass(pass_id, timeout=DISPATCH_CHECK_READ_TIMEOUT_SECONDS)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its row before polling its "
            "batch; dispatching as before",
            exc_info=True,
        )
        return None


async def _finalize_stuck_duplicate(bp: BatchPass, ops: DreamOperations) -> None:
    """On a duplicate delivery, finalize the admin job row iff the first
    delivery crashed between apply and ``mark_complete`` and left it
    non-terminal. An already-terminal row is left untouched so the first
    delivery's real apply stats are never overwritten.

    Best-effort: a status read/write failure here must not crash the
    duplicate tail (lock release + cleanup still need to run)."""
    if not bp.job_id:
        return
    try:
        existing = await job_status.read_status(kind="dream_pass", job_id=bp.job_id)
        if existing is None or existing.state in ("complete", "errored"):
            return
        logger.warning(
            "Duplicate dispatch found job %s stuck in state=%s — "
            "finalizing with the clamped op counts",
            bp.job_id[:12],
            existing.state,
        )
        await job_status.mark_complete(
            kind="dream_pass", job_id=bp.job_id, result=_attempted_result(bp, ops)
        )
    except Exception:
        logger.exception(
            "Failed to finalize stuck duplicate for job %s", bp.job_id[:12]
        )


def _attempted_result(bp: BatchPass, ops: DreamOperations) -> DreamPassResult:
    """The first delivery's per-edge outcomes (and dream session id) died
    with it, so the counts here are the clamped *attempted* ops — annotated
    so the admin UI doesn't present them as confirmed apply results."""
    note = (
        "[finalized after duplicate delivery — counts reflect attempted "
        "operations; writes landed with the original delivery] "
    )
    return DreamPassResult(
        user_id=bp.user_id,
        pass_id=bp.pass_id,
        execution_path="anthropic_batch",
        consolidated_count=len(ops.writes),
        proposal_count=len(ops.proposals),
        demotion_count=len(ops.demotions),
        entity_invalidation_count=len(ops.entity_invalidations),
        # Batch apply never drains in-line by design; the first
        # delivery's writes (if any) landed fire-and-forget. ``skipped``
        # marks this as a healthy by-design skip, NOT a drain failure.
        ingestion_drain_status=IngestionDrainStatus.skipped,
        summary_for_user=note + (ops.summary_for_user or ""),
    )
