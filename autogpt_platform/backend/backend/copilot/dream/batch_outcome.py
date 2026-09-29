"""How a batch dream pass ends: what it reports to the admin JobStatus row and
to its durable DreamPass record, and the cleanup after it.

A batch pass ends in a callback, in another process, long after it started:
``fail_pass`` for every failure, ``record_completion`` once apply has run.
Either way the record gets the usage of every phase that landed, read off the
pass's Redis state and priced in ``batch_costs.py``, unless a stop closed the
record first: a closed record refuses the write, and that usage lives on only
in the cost log. Either way the record is marked for the cleanup after the
pass, which ``clean_up_after`` then does (``cleanup.py``).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.copilot.graphiti.scope import MemoryScope

from .batch_costs import landed_usage
from .batch_state import read_state_or_none
from .batch_submit import read_lock_token
from .cleanup import clean_up_pass, finish_cleanup
from .locks import release_dream_lock
from .schemas import (
    DreamOperationsSnapshot,
    DreamPassResult,
    DreamPassUsage,
    IngestionDrainStatus,
)
from .store import read_pass, record_batch_complete, record_batch_failed

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


async def fail_pass(
    bp: BatchPass,
    error: str,
    *,
    holds_lock: bool = True,
    provider_batch_id: str | None = None,
) -> None:
    """Close the pass errored: its admin job and its record first, the
    record marked for the cleanup after it, then that cleanup
    (``clean_up_after``): the batch *provider_batch_id* names stopped at the
    provider, the phases that landed charged, the lock released (unless the
    pass lost it, *holds_lock* false: it is then another pass's) and the
    per-pass state deleted. Each step is best-effort on its own, so one that
    fails never stops the ones after it, and leaves the row marked for the
    reaper.

    The record and the job's result carry the usage of the phases that
    landed, read off the pass's state (a stop that closed the record first
    keeps it off the record, not off the job); a state that cannot be read
    leaves that usage unknown rather than the record open.

    We incurred the provider tokens for completed phases regardless of
    whether the whole pass landed, so they're recorded against the
    user's usage — matching the sync path and the documented contract in
    ``dream/billing.py``. The per-phase claims inside
    ``charge_landed_phases`` keep this at-most-once.
    """
    logger.warning("Dream batch pass=%s failed: %s", bp.pass_id, error)
    state = await read_state_or_none(bp.pass_id)
    usage = landed_usage(state, bp.phase_models, bp.pass_id)
    await mark_job_errored(bp.job_id, error, result=_failed_result(bp, error, usage))
    await record_batch_failed(bp.pass_id, error, usage)
    await clean_up_after(bp, holds_lock=holds_lock, provider_batch_id=provider_batch_id)


async def clean_up_after(
    bp: BatchPass, *, holds_lock: bool = True, provider_batch_id: str | None = None
) -> None:
    """The cleanup after the ended pass (``cleanup.clean_up_pass``), its lock
    released under the pass's token (``lock_token_of``), then its row's mark
    cleared if every step finished; a step that did not, a lock the pass has
    no token for included, leaves the mark for the reaper. Never raises: on
    the success tail an exception would fire AFTER ``mark_complete`` and the
    crash guard would rewrite a completed job to errored."""
    try:
        scope = MemoryScope.build(bp.user_id, bp.expert_id)
    except ValueError:
        logger.warning(
            f"Dream pass {bp.pass_id}: invalid memory scope for user "
            f"{bp.user_id[:12]}; its cleanup is left for the reaper"
        )
        return
    cleanup = await clean_up_pass(
        bp.pass_id,
        scope,
        phase_models=bp.phase_models,
        provider_batch_id=provider_batch_id,
        lock_token=await lock_token_of(bp.pass_id) if holds_lock else None,
        release=holds_lock,
    )
    await finish_cleanup(bp.pass_id, cleanup)


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


async def mark_job_errored(
    job_id: str,
    error: str,
    *,
    dead_end: bool = False,
    result: DreamPassResult | None = None,
) -> None:
    """Close the admin job row errored, with the pass's *result* when there
    is one (its usage). Best-effort: status write failures are logged, never
    raised; a *dead_end* (a payload no phase handler can take) logs under its
    own message."""
    if not job_id:
        return
    try:
        from .job_status import mark_errored

        await mark_errored(kind="dream_pass", job_id=job_id, error=error, result=result)
    except Exception:
        if dead_end:
            logger.exception("Failed to mark dead-end job %s errored", job_id[:12])
        else:
            logger.exception("Failed to mark dream pass job %s errored", job_id)


async def release_lock(bp: BatchPass) -> bool:
    """Release the disowned dream lock under the pass's ownership token
    (``lock_token_of``), and say whether the lock is no longer the pass's
    (``release_dream_lock``). Must run before ``delete_input_bundle``, whose
    key carries the token first. With no token anywhere the lock is left
    alone rather than blind-deleting what may be a newer pass's lock.

    Best-effort like ``release_dream_lock`` itself: a Redis blip on the
    token read must not propagate — on the success tail it would fire
    AFTER ``mark_complete`` and the crash guard would rewrite a completed
    job to errored."""
    try:
        scope = MemoryScope.build(bp.user_id, bp.expert_id)
    except ValueError:
        logger.warning(
            "Invalid memory scope for disowned dream lock of user %s — "
            "leaving it for the TTL to clear",
            bp.user_id[:12],
        )
        return False
    return await release_dream_lock(scope, await lock_token_of(bp.pass_id))


async def lock_token_of(pass_id: str) -> str | None:
    """The token the batch pass holds its scope's lock under: the one its
    input bundle carries, else the one its row keeps (one bounded read, for
    a bundle that lost its token or is gone). ``None`` when neither can say:
    the pass's ownership of the lock is then unknown, never taken for "no
    lock", so it neither applies nor has its lock taken for released."""
    if not pass_id:
        return None
    token = await _bundle_lock_token(pass_id)
    if token is not None:
        return token
    token = await _row_lock_token(pass_id)
    if token is not None:
        logger.warning(
            f"Dream pass {pass_id}: no lock token with its input bundle; "
            "going by the one its row keeps"
        )
    return token


async def _bundle_lock_token(pass_id: str) -> str | None:
    """The lock token the pass's input bundle carries; ``None`` when there is
    no bundle, it carries none, or Redis cannot say."""
    try:
        return await read_lock_token(pass_id)
    except Exception:
        logger.exception(
            "Failed to read dream lock token for pass=%s — going by its row",
            pass_id,
        )
        return None


async def _row_lock_token(pass_id: str) -> str | None:
    """The lease token the pass's row keeps, read once under the store's
    deadline; ``None`` when there is none, or the store cannot say in time."""
    try:
        row = await read_pass(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its row for its lock token",
            exc_info=True,
        )
        return None
    return row.lease_token if row is not None else None


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
        dropped_forgotten=_stat_count(apply_stats, "dropped_forgotten"),
        uncited_writes_dropped=_stat_count(apply_stats, "uncited_writes_dropped"),
        cross_scope_citations_dropped=_stat_count(
            apply_stats, "cross_scope_citations_dropped"
        ),
        protected_demotions=_stat_count(apply_stats, "protected_demotions"),
        indeterminate_demotion_writes=_stat_count(
            apply_stats, "indeterminate_demotion_writes"
        ),
        # Only apply's own False marks the count unconfirmed.
        demotion_accounting_complete=(
            apply_stats.get("demotion_accounting_complete") is not False
        ),
        dream_session_id=raw_session_id if isinstance(raw_session_id, str) else None,
        ingestion_drain_status=ingestion_drain_status,
        operations=_stat_snapshot(apply_stats),
        # Carry the user-facing narrative like the sync path does —
        # without it the Memory Visualizer renders a blank summary for
        # batch-completed dreams even though the session message exists.
        summary_for_user=summary_for_user,
    )


def _failed_result(
    bp: BatchPass, error: str, usage: DreamPassUsage | None
) -> DreamPassResult:
    """A failed batch pass as its job reports it: the error, and what the
    phases that landed used."""
    return DreamPassResult(
        user_id=bp.user_id,
        pass_id=bp.pass_id,
        execution_path="anthropic_batch",
        error=error,
        usage=usage,
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
