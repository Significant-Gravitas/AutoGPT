"""How a batch dream pass ends: what it reports to the admin JobStatus row and
to its durable DreamPass record, and the lock it gives back.

A batch pass ends in a callback, in another process, long after it started:
``fail_pass`` for every failure, ``record_completion`` once apply has run.
Either way the record gets the usage of every phase that landed, read off the
pass's Redis state and priced in ``batch_costs.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.copilot.graphiti.scope import MemoryScope

from .batch_costs import landed_usage, log_all_phase_costs
from .batch_state import best_effort_cleanup, read_state
from .batch_submit import read_lock_token
from .locks import release_dream_lock
from .schemas import (
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassResult,
    DreamPassUsage,
    IngestionDrainStatus,
)
from .store import record_batch_complete, record_batch_failed

logger = logging.getLogger(__name__)

# What ``apply.apply_operations`` reports.
ApplyStats = dict[str, int | str | IngestionDrainStatus | DreamOperationsSnapshot]


class BatchPass(BaseModel):
    """One batch pass as a callback sees it: whose it is, the admin job it
    reports to (empty when there is none), and the model each phase ran on."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    expert_id: str | None
    pass_id: str
    job_id: str
    phase_models: dict[str, str]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BatchPass":
        """The pass a pending entry's payload names. Missing fields read as
        empty; the handler turns those into a dead end."""
        expert_id = payload.get("expert_id")
        return cls(
            user_id=str(payload.get("user_id") or ""),
            expert_id=str(expert_id) if expert_id is not None else None,
            pass_id=str(payload.get("pass_id") or ""),
            job_id=str(payload.get("job_id") or ""),
            phase_models=_phase_models(payload.get("phase_models")),
        )


async def fail_pass(bp: BatchPass, error: str) -> None:
    """Mark JobStatus errored, record usage for any phases that already
    landed, close the pass's record, then release the lock and clean up
    per-pass state.

    We incurred the provider tokens for completed phases regardless of
    whether the whole pass landed, so they're recorded against the
    user's usage — matching the sync path and the documented contract in
    ``dream/billing.py``. The idempotency gate inside
    ``log_all_phase_costs`` keeps this at-most-once.
    """
    logger.warning("Dream batch pass=%s failed: %s", bp.pass_id, error)
    await mark_job_errored(bp.job_id, error)
    state = await read_state(bp.pass_id)
    if state:
        await log_all_phase_costs(
            user_id=bp.user_id,
            expert_id=bp.expert_id,
            pass_id=bp.pass_id,
            state=state,
            phase_models=bp.phase_models,
        )
    await record_batch_failed(
        bp.pass_id, error, landed_usage(state, bp.phase_models, bp.pass_id)
    )
    # Release the dream lock the batch path disowned to this callback.
    await release_lock(bp)
    await best_effort_cleanup(bp.pass_id)


async def record_completion(
    bp: BatchPass,
    apply_stats: ApplyStats,
    *,
    ingestion_drain_status: IngestionDrainStatus,
    summary_for_user: str,
    usage: DreamPassUsage | None,
) -> None:
    """Close the admin job and the pass's record with what apply reported.

    Neither may raise: apply's writes have landed, and the crash guard would
    route an exception here to ``fail_pass`` and report the pass errored.
    """
    try:
        pass_result = _applied_result(
            bp, apply_stats, ingestion_drain_status, summary_for_user
        )
    except Exception:
        logger.exception("Failed to build the result of dream pass %s", bp.pass_id)
        return
    if bp.job_id:
        try:
            from .job_status import mark_complete

            await mark_complete(kind="dream_pass", job_id=bp.job_id, result=pass_result)
        except Exception:
            logger.exception("Failed to mark dream pass job %s complete", bp.job_id)
    await record_batch_complete(pass_result, usage)


async def finalize_stuck_duplicate(bp: BatchPass, ops: DreamOperations) -> None:
    """On a duplicate delivery, finalize the admin job row iff the first
    delivery crashed between apply and ``mark_complete`` and left it
    non-terminal. An already-terminal row is left untouched so the first
    delivery's real apply stats are never overwritten.

    Best-effort: a status read/write failure here must not crash the
    duplicate tail (lock release + cleanup still need to run)."""
    if not bp.job_id:
        return
    try:
        from .job_status import mark_complete, read_status

        existing = await read_status(kind="dream_pass", job_id=bp.job_id)
        if existing is None or existing.state in ("complete", "errored"):
            return
        logger.warning(
            "Duplicate dispatch found job %s stuck in state=%s — "
            "finalizing with the clamped op counts",
            bp.job_id[:12],
            existing.state,
        )
        await mark_complete(
            kind="dream_pass", job_id=bp.job_id, result=_attempted_result(bp, ops)
        )
    except Exception:
        logger.exception(
            "Failed to finalize stuck duplicate for job %s", bp.job_id[:12]
        )


async def mark_job_errored(job_id: str, error: str) -> None:
    """Close the admin job row errored. Best-effort: status write failures
    are logged, never raised."""
    if not job_id:
        return
    try:
        from .job_status import mark_errored

        await mark_errored(kind="dream_pass", job_id=job_id, error=error)
    except Exception:
        logger.exception("Failed to mark dream pass job %s errored", job_id[:12])


async def release_lock(bp: BatchPass) -> None:
    """Release the disowned dream lock with the ownership token persisted
    alongside the input bundle. Must run before ``delete_input_bundle`` —
    the token rides on that key. A missing token (bundle TTL'd out,
    malformed payload) leaves the lock for its TTL to clear rather than
    blind-deleting what may be a newer pass's lock.

    Best-effort like ``release_dream_lock`` itself: a Redis blip on the
    token read must not propagate — on the success tail it would fire
    AFTER ``mark_complete`` and the crash guard would rewrite a completed
    job to errored. Falls back to a token-less release (lock TTL)."""
    token: str | None = None
    if bp.pass_id:
        try:
            token = await read_lock_token(bp.pass_id)
        except Exception:
            logger.exception(
                "Failed to read dream lock token for pass=%s — "
                "leaving the lock to its TTL",
                bp.pass_id,
            )
    try:
        scope = MemoryScope.build(bp.user_id, bp.expert_id)
    except ValueError:
        logger.warning(
            "Invalid memory scope for disowned dream lock of user %s — "
            "leaving it for the TTL to clear",
            bp.user_id[:12],
        )
        return
    await release_dream_lock(scope, token)


def _applied_result(
    bp: BatchPass,
    apply_stats: ApplyStats,
    ingestion_drain_status: IngestionDrainStatus,
    summary_for_user: str,
) -> DreamPassResult:
    """The ``DreamPassResult`` of a batch pass whose apply just returned."""
    raw_session_id = apply_stats.get("session_id")
    return DreamPassResult(
        user_id=bp.user_id,
        pass_id=bp.pass_id,
        execution_path="anthropic_batch",
        consolidated_count=_stat_count(apply_stats, "consolidated_count"),
        proposal_count=_stat_count(apply_stats, "proposal_count"),
        demotion_count=_stat_count(apply_stats, "demotion_count"),
        entity_invalidation_count=_stat_count(apply_stats, "entity_invalidation_count"),
        dream_session_id=raw_session_id if isinstance(raw_session_id, str) else None,
        ingestion_drain_status=ingestion_drain_status,
        operations=_stat_snapshot(apply_stats),
        # Carry the user-facing narrative like the sync path does —
        # without it the Memory Visualizer renders a blank summary for
        # batch-completed dreams even though the session message exists.
        summary_for_user=summary_for_user,
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


def _stat_count(apply_stats: ApplyStats, key: str) -> int:
    """One apply count as a plain ``int``, 0 when missing or malformed: the
    stats values are a union that includes the snapshot."""
    value = apply_stats.get(key)
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _stat_snapshot(apply_stats: ApplyStats) -> DreamOperationsSnapshot | None:
    raw_snapshot = apply_stats.get("snapshot")
    if isinstance(raw_snapshot, DreamOperationsSnapshot):
        return raw_snapshot
    if isinstance(raw_snapshot, dict):
        return DreamOperationsSnapshot.model_validate(raw_snapshot)
    return None


def _phase_models(raw: Any) -> dict[str, str]:
    """Per-phase model map persisted by ``submit_phase`` — used to chain
    the next phase and to price each phase with the model it actually
    used. Empty when absent (current code never omits it)."""
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}
