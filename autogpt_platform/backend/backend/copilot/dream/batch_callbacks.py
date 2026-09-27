"""Dream-pass batch result handler — sequential phase chain.

Registered with the BatchExecutor service under namespace
``"dream_pass"`` at module-import time. When a batch the dream pass
submitted finishes, the BatchExecutor calls
``handle_dream_batch_result`` with the per-``custom_id`` result rows
plus the pending entry's ``payload``.

Dream's three phases are sequentially dependent:

  consolidate → recombine (needs consolidate output)
              → sanitize  (needs both)

That means a phase can only be submitted once the prior phase's
result lands. The orchestrator kicks off phase 1; this handler chains
phase 2 from phase 1's result, phase 3 from phase 2's result, then
runs the apply step + cost log + JobStatus complete when phase 3 lands.

The pass's Redis state and its at-most-once gates are ``batch_state.py``;
how the pass ends (JobStatus, the durable ``DreamPass`` record, the lock)
is ``batch_outcome.py``; what its phases cost and used is ``batch_costs.py``;
the deliveries that never run the chain (a closed pass's batch, a dead-end
payload, a duplicate) are ``batch_deliveries.py``. Each callback advances the
record and, once its phase has landed, renews the pass's lease (``lease.py``),
ending a pass whose lock is no longer its own. It ends a pass cancelled or
expired meanwhile before it chains or claims the apply gate (``cancel.py``),
and after the claim applies only on a renewal proving the lock is the pass's
(``lease.admit_batch_apply``).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.inference.context import InferenceError

from .batch_costs import landed_usage
from .batch_deliveries import (
    drop_closed,
    end_dead_end,
    finish_duplicate,
    should_dispatch,
)
from .batch_outcome import (
    ApplyStats,
    BatchPass,
    clean_up_after,
    fail_pass,
    record_completion,
    release_lock,
)
from .batch_state import claim_apply_gate, content_for, read_state, write_phase_to_state
from .batch_submit import (
    PHASE_RESPONSE_MODELS,
    read_input_bundle,
    read_lock_token,
    submit_phase,
)
from .cancel import end_batch_pass_if_stopped, pass_closed
from .clamp import clamp_operations
from .lease import ApplyLease, admit_batch_apply, renew_batch_lease
from .llm import parse_json_with_prose_fallback
from .provider_batch import anthropic_api_key
from .schemas import DreamOperations, DreamPhase, IngestionDrainStatus
from .store import record_applying, record_next_batch, record_phase_output

if TYPE_CHECKING:
    from backend.executor.batch_executor import PendingEntry
    from backend.util.llm.providers import BatchResultRow

    from .fetch import DreamInput

logger = logging.getLogger(__name__)


NAMESPACE = "dream_pass"

NEXT_PHASE: dict[DreamPhase, DreamPhase | None] = {
    "consolidate": "recombine",
    "recombine": "sanitize",
    "sanitize": None,  # terminal — apply runs from here
}


async def handle_dream_batch_result(
    entry: PendingEntry, rows: list[BatchResultRow]
) -> None:
    """BatchExecutor entry, once per finished phase batch: validate and keep
    the phase's result, then end the pass on an error, submit the next phase
    (the job left ``submitted`` at it), or after sanitize apply, charge the
    three phases and complete the job. A payload no phase can take is a dead
    end (``batch_deliveries``)."""
    payload = entry.payload or {}
    bp = BatchPass.from_payload(payload)
    phase = payload.get("phase")
    if not bp.user_id or not bp.pass_id or not phase:
        logger.warning(
            "Dream batch handler missing user_id/pass_id/phase — payload=%s",
            payload,
        )
        await end_dead_end(bp, "batch payload missing user_id/pass_id/phase")
        return
    if phase not in NEXT_PHASE:
        logger.warning("Dream batch handler unknown phase=%r", phase)
        await end_dead_end(bp, f"unknown batch phase {phase!r}")
        return
    await _handle_guarded(bp, phase, rows)


async def _handle_guarded(
    bp: BatchPass, phase: DreamPhase, rows: list[BatchResultRow]
) -> None:
    """Check the payload's scope against the pass's own, then handle the
    phase, all inside one crash guard.

    The batch path disowned the per-user dream lock to this callback, and
    BatchExecutor._dispatch swallows handler exceptions — so without this
    guard an unexpected error here would strand the user behind the
    disowned lock until its extended TTL expired and leave the admin
    JobStatus row stuck. The crash goes through ``fail_pass`` (releases the
    lock + marks the job errored); if even that fails, the lock is released
    directly so the user is never blocked on a leaked lock.
    """
    scoped = bp
    try:
        input_bundle = await read_input_bundle(bp.pass_id)
        if input_bundle is None and await pass_closed(bp.pass_id):
            logger.debug("Dream pass %s ended; late delivery ignored", bp.pass_id)
            return
        if input_bundle is None:
            logger.error(
                "Dream batch input missing; refusing payload-only scope for pass=%s",
                bp.pass_id,
            )
            await fail_pass(bp, "batch DreamInput missing; memory scope unavailable")
            return
        scoped = bp.model_copy(
            update={
                "user_id": input_bundle.user_id,
                "expert_id": input_bundle.expert_id,
            }
        )
        if scoped != bp:
            logger.error(
                "Dream batch payload memory scope mismatch for pass=%s", bp.pass_id
            )
            await fail_pass(scoped, "batch payload memory scope mismatch")
            return
        await _handle_phase_result(bp, input_bundle, phase, rows)
    except Exception:
        logger.exception(
            "Dream batch handler crashed for pass=%s phase=%s", bp.pass_id, phase
        )
        await _fail_after_crash(scoped, f"{phase}: handler crashed")


async def _fail_after_crash(bp: BatchPass, error: str) -> None:
    try:
        await fail_pass(bp, error)
    except Exception:
        logger.exception("Dream batch _fail_pass also failed for pass=%s", bp.pass_id)
        try:
            await release_lock(bp)
        except Exception:
            logger.exception(
                "Dream batch lock release failed for user=%s", bp.user_id[:12]
            )


async def _handle_phase_result(
    bp: BatchPass,
    input_bundle: DreamInput,
    phase: DreamPhase,
    rows: list[BatchResultRow],
) -> None:
    """Validate one finished phase batch, then chain to the next phase or
    finalize. Every early return below finalizes via ``fail_pass``; an
    unexpected raise (Redis blip, apply bug) is the crash guard's."""
    if not rows:
        logger.warning("Dream batch handler got empty rows for pass=%s", bp.pass_id)
        await fail_pass(bp, f"{phase}: provider returned no rows")
        return
    # Single-request-per-batch today; the first (and only) row is the
    # phase result. When we group batches in the future the BatchExecutor
    # will already split by custom_id before calling us.
    if await _landed_output(bp, phase, rows[0]) is None:
        return
    next_phase = NEXT_PHASE[phase]
    if not await renew_batch_lease(bp, next_phase or "apply"):
        return
    if next_phase is not None:
        await _chain_next_phase(bp, input_bundle, next_phase)
        return
    # Terminal phase landed — apply + finalize.
    await _finalize_complete(bp, input_bundle)


async def _landed_output(
    bp: BatchPass, phase: DreamPhase, row: BatchResultRow
) -> BaseModel | None:
    """The phase's validated output, kept in the pass's Redis state and its
    record; ``None`` once an errored or malformed row has failed the pass."""
    if row.error:
        await write_phase_to_state(pass_id=bp.pass_id, phase=phase, row=row)
        await fail_pass(bp, f"{phase}: {row.error}")
        return None
    # Validate the row's content matches the phase's Pydantic schema
    # BEFORE persisting — corrupted content shouldn't pollute the
    # accumulator for the next phase to read back. Parsed the way the sync
    # path parses it: a model the output tool couldn't be forced on may
    # answer in text, its JSON fenced or behind prose, so what is stored
    # (and read back by the next phase) is the JSON alone.
    try:
        payload = parse_json_with_prose_fallback(row.content)
        output = PHASE_RESPONSE_MODELS[phase].model_validate(payload)
    except (InferenceError, ValidationError) as exc:
        await write_phase_to_state(pass_id=bp.pass_id, phase=phase, row=row)
        await fail_pass(bp, f"{phase}: invalid output shape — {type(exc).__name__}")
        return None
    await write_phase_to_state(
        pass_id=bp.pass_id, phase=phase, row=row.with_content(json.dumps(payload))
    )
    await record_phase_output(bp.pass_id, phase, output)
    return output


async def _chain_next_phase(
    bp: BatchPass, input_bundle: DreamInput, next_phase: DreamPhase
) -> None:
    """Submit the next phase in the chain, unless the pass was stopped.

    Uses the validated ``DreamInput`` + accumulated prior phase outputs
    from Redis, builds the next phase's prompt, fires another
    batch submission. On any failure to submit, marks the JobStatus
    errored — silent submission failures are unrecoverable.
    """
    if await end_batch_pass_if_stopped(bp):
        return
    state = await read_state(bp.pass_id)
    api_key = anthropic_api_key()
    if api_key is None:
        await fail_pass(bp, f"{next_phase}: no Anthropic API key configured")
        return
    try:
        submission = await submit_phase(
            user_id=bp.user_id,
            pass_id=bp.pass_id,
            job_id=bp.job_id,
            phase=next_phase,
            phase_models=bp.phase_models,
            api_key=api_key,
            input_bundle=input_bundle,
            consolidated_json=content_for(state, "consolidate"),
            recombined_json=content_for(state, "recombine"),
        )
    except Exception as exc:
        logger.exception(
            "Failed to submit %s phase for pass=%s — marking errored",
            next_phase,
            bp.pass_id,
        )
        await fail_pass(bp, f"{next_phase}: submit failed: {type(exc).__name__}: {exc}")
        return
    await _advance_job(bp, next_phase, submission.provider_batch_id)
    await record_next_batch(bp.pass_id, next_phase, submission.provider_batch_id)


async def _advance_job(
    bp: BatchPass, next_phase: DreamPhase, provider_batch_id: str
) -> None:
    """Point the admin job row at the next phase's batch; best-effort."""
    if not bp.job_id:
        return
    try:
        from .job_status import update_status_phase

        await update_status_phase(
            kind="dream_pass",
            job_id=bp.job_id,
            state="submitted",
            current_phase=next_phase,
            batch_id=provider_batch_id,
        )
    except Exception:
        logger.exception(
            "Failed to update status for next phase=%s pass=%s",
            next_phase,
            bp.pass_id,
        )


async def _finalize_complete(bp: BatchPass, input_bundle: DreamInput) -> None:
    """Sanitize phase has landed. Run apply + cost log + complete."""
    try:
        from .apply import (
            BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS,
            apply_operations,
            drain_status_from_stats,
        )
    except Exception:
        logger.exception("Failed to import dream apply for pass=%s", bp.pass_id)
        await fail_pass(bp, "apply: import failed")
        return
    state = await read_state(bp.pass_id)
    ops = await _terminal_ops(bp, state, input_bundle)
    lease = await _claim_apply(bp, ops) if ops is not None else None
    if ops is None or lease is None:
        return
    try:
        # No ingestion drain here: walk_once awaits this handler serially, so
        # an in-line drain would stall every other user's pending batch
        # (``BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS``).
        apply_stats = await apply_operations(
            MemoryScope.build(bp.user_id, bp.expert_id),
            bp.pass_id,
            ops,
            known_fact_uuids=input_bundle.known_fact_uuids,
            ingestion_drain_timeout=BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS,
            lease=lease,
        )
    except Exception as exc:
        logger.exception(
            "apply_operations crashed for batch pass=%s — marking errored",
            bp.pass_id,
        )
        await fail_pass(bp, f"apply: {type(exc).__name__}: {exc}")
        return
    await _finish_applied(
        bp, state, apply_stats, ops, drain_status_from_stats(apply_stats)
    )


async def _claim_apply(bp: BatchPass, ops: DreamOperations) -> ApplyLease | None:
    """Record APPLYING, check for a stop, claim the apply gate, admit apply
    on a renewal of the lock: the lease apply renews again before it writes,
    or ``None`` once a stop, a duplicate, an unreadable gate or an unproven
    lease has ended this delivery. A delivery that dies before the claim has
    not claimed it, so a redelivery still applies. After the claim only the
    compare-and-extend runs, so a newer pass that took the scope meanwhile is
    never applied over (``lease.py`` has what stays open)."""
    await record_applying(bp.pass_id, ops)
    lock_token = await read_lock_token(bp.pass_id)
    if await end_batch_pass_if_stopped(bp):
        return None
    gate = await claim_apply_gate(bp.pass_id)
    if gate == "error":
        # We cannot tell first-vs-duplicate apart, and "complete with no
        # writes" would silently drop the dream.
        await fail_pass(
            bp, "apply: gate unavailable (redis) — cannot guarantee at-most-once"
        )
        return None
    if gate == "duplicate":
        await finish_duplicate(bp, ops)
        return None
    return await admit_batch_apply(bp, lock_token)


async def _terminal_ops(
    bp: BatchPass, state: dict[str, dict[str, Any]], input_bundle: DreamInput
) -> DreamOperations | None:
    """The sanitize phase's operations, clamped; ``None`` once a missing or
    malformed result has failed the pass."""
    sanitize_row = state.get("sanitize")
    if sanitize_row is None or not sanitize_row.get("content"):
        await fail_pass(bp, "sanitize: missing terminal phase content")
        return None
    try:
        ops = DreamOperations.model_validate(json.loads(sanitize_row["content"]))
    except (json.JSONDecodeError, ValidationError) as exc:
        await fail_pass(bp, f"sanitize: shape validation failed: {type(exc).__name__}")
        return None
    # Enforce the same per-pass operation caps the sync path applies
    # before writing — the model can over-emit past the prompt's limits.
    # The 5%-of-active-facts demotion ceiling needs the original fact
    # count, and the known-fact allowlist filters hallucinated demotion
    # uuids BEFORE the cap slice (else they displace valid demotions).
    return clamp_operations(
        ops,
        len(input_bundle.facts),
        known_fact_uuids=input_bundle.known_fact_uuids,
    )


async def _finish_applied(
    bp: BatchPass,
    state: dict[str, dict[str, Any]],
    apply_stats: ApplyStats,
    ops: DreamOperations,
    ingestion_drain_status: IngestionDrainStatus,
) -> None:
    """Close the job and the record, marked, then clean up after the pass
    (``clean_up_after``): the landed phases charged, the lock the batch path
    disowned to this callback released so the user's next dream can run.
    Failure paths do the same through ``fail_pass``; the per-phase claims
    keep the charges at-most-once across every path."""
    # The batch path skips the drain by design, so apply reports
    # ``skipped`` whenever the pass enqueued writes (``drained`` only
    # for an empty pass), read via the shared, fail-closed helper.
    await record_completion(
        bp,
        apply_stats,
        ingestion_drain_status=ingestion_drain_status,
        summary_for_user=ops.summary_for_user,
        usage=landed_usage(state, bp.phase_models, bp.pass_id),
    )
    await clean_up_after(bp)


def _register() -> None:
    try:
        from backend.executor.batch_executor import register_handler

        register_handler(
            NAMESPACE,
            handle_dream_batch_result,
            should_dispatch=should_dispatch,
            on_drop=drop_closed,
        )
    except Exception:
        logger.exception(
            "Failed to register dream batch handler — "
            "batch results will not dispatch"
        )


_register()
