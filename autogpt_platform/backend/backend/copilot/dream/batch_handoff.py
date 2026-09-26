"""Hands a gathered dream pass to Anthropic's Message Batches API.

The sync entry point runs this when routing picked ``anthropic_batch``: it
persists the input bundle, submits phase 1 (consolidate) and returns at once.
The BatchExecutor polls; ``batch_callbacks`` submits phases 2 (recombine) and
3 (sanitize) as each prior result lands, then applies and closes the pass.
The dream lock goes with the pass: extended to the batch window here and
released by the callbacks when the pass ends.

Any caller can take this route (it depends on the flag and the deployment's
key, not the caller). ``status_id`` ties the callbacks' updates to the
JobStatus row an admin trigger created; without one the callbacks skip the
JobStatus writes, and apply and cost logging still run.
"""

import logging

from backend.copilot.config import ChatConfig
from backend.executor.batch_executor import remove_pending

from .batch_submit import (
    delete_input_bundle,
    persist_input_bundle,
    phase_models_for_config,
    submit_phase,
)
from .fetch import DreamInput
from .locks import BATCH_LOCK_TTL_SECONDS, DreamLockHandle
from .pass_run import DreamPassRun
from .schemas import DreamPassResult
from .store import record_submitted

logger = logging.getLogger(__name__)


async def submit_dream_pass_batch(
    run: DreamPassRun,
    *,
    config: ChatConfig,
    input_bundle: DreamInput,
    status_id: str | None,
    lock_handle: DreamLockHandle,
) -> DreamPassResult:
    """Submit phase 1 and hand the pass, and its lock, to the callbacks."""
    api_key = config.direct_anthropic_api_key
    if not api_key:
        # Shouldn't happen — routing.py only picks anthropic_batch when
        # the key is present. Guard for type-narrowing + future safety.
        return run.failure("anthropic_batch: no Anthropic API key (routing bug)")

    # Persist DreamInput so the per-phase callbacks can rebuild the
    # next phase's prompt without re-fetching from Postgres + FalkorDB.
    try:
        await persist_input_bundle(
            run.pass_id, input_bundle, lock_token=lock_handle.token
        )
    except Exception as exc:
        return run.failure(f"anthropic_batch: input persist failed: {exc}")

    # An empty job id tells the callbacks there is no JobStatus row to update.
    try:
        submission = await submit_phase(
            user_id=run.user_id,
            pass_id=run.pass_id,
            job_id=status_id or "",
            phase="consolidate",
            phase_models=phase_models_for_config(config),
            api_key=api_key,
            input_bundle=input_bundle,
        )
    except Exception as exc:
        return run.failure(f"anthropic_batch: phase 1 submit failed: {exc}")
    return await _hand_off(run, lock_handle, submission.provider_batch_id, input_bundle)


async def _hand_off(
    run: DreamPassRun,
    lock_handle: DreamLockHandle,
    provider_batch_id: str,
    input_bundle: DreamInput,
) -> DreamPassResult:
    """Phase 1 is enqueued — hand the dream lock to the batch callback so it
    spans the full async lifetime (apply runs hours later). Extend the TTL to
    the batch window first; the callback releases it on terminal/failure.

    A failed extend means the lock expired before the handoff — a newer pass
    may already own the graph, so the just-submitted batch must never be
    applied: revoke the pending entry (the poller then never dispatches the
    callback chain) and drop the input bundle. The provider batch is orphaned;
    its results are discarded. The lock is NOT disowned, so the context
    manager's compare-and-delete release stays a safe no-op.
    """
    logger.info(
        "Dream pass %s submitted via Anthropic batch=%s (phase=consolidate)",
        run.pass_id,
        provider_batch_id,
    )
    if not await lock_handle.extend(BATCH_LOCK_TTL_SECONDS):
        await remove_pending(provider_batch_id)
        await delete_input_bundle(run.pass_id)
        return run.failure(
            "anthropic_batch: dream lock lost before handoff — batch revoked"
        )
    lock_handle.disown()
    await record_submitted(
        run.pass_id,
        input_bundle=input_bundle,
        provider_batch_id=provider_batch_id,
        lease_token=lock_handle.token,
        lease_ttl_seconds=BATCH_LOCK_TTL_SECONDS,
    )
    return run.handed_off()
