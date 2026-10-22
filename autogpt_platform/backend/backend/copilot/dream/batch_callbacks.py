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
from backend.copilot.inference.parse_output import parse_json_with_prose_fallback
from backend.copilot.trial_cost_context import TrialCostContext, restore_cost_context
from backend.copilot.usage_activation import UsageActivationUnavailable

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
    lock_token_of,
    record_completion,
    release_lock,
)
from .batch_state import claim_apply_gate, content_for, read_state, write_phase_to_state
from .batch_submit import PHASE_RESPONSE_MODELS, read_input_bundle, submit_phase
from .cancel import end_batch_pass_if_stopped, pass_closed
from .citations import fact_scopes
from .clamp import clamp_pass_operations
from .lease import ApplyLease, admit_batch_apply, renew_batch_lease
from .locks import DreamLockLostError
from .pass_errors import pass_error
from .provider_batch import anthropic_api_key
from .schemas import DreamOperations, DreamPhase, IngestionDrainStatus
from .store import (
    read_cost_context,
    record_applying,
    record_next_batch,
    record_phase_output,
)

if TYPE_CHECKING:
    from backend.executor.batch_executor import PendingEntry
    from backend.util.llm.providers import BatchResultRow

    from .fetch import DreamInput

logger = logging.getLogger(__name__)


NAMESPACE = "dream_pass"

# The error a batch pass closes with when the usage attribution it started
# under could not be established for one of its results (#15208): a fixed
# code (``pass_errors``). Its phases are charged only under the attribution
# its row carries, if any (``batch_costs``).
UNATTRIBUTED_ERROR = pass_error("batch", "usage_attribution_unavailable")

NEXT_PHASE: dict[DreamPhase, DreamPhase | None] = {
    "consolidate": "recombine",
    "recombine": "sanitize",
    "sanitize": None,  # terminal — apply runs from here
}


async def handle_dream_batch_result(
    entry: PendingEntry, rows: list[BatchResultRow]
) -> None:
    """BatchExecutor entry, once per finished phase batch: the usage
    attribution the pass started under (#15208), read off its row
    (``_pass_attribution``), restored around the phase's handling
    (``_handle_dream_batch_result``); a pass without one, or whose batch
    carries another, ends (``_fail_unattributed_pass``)."""
    payload = entry.payload or {}
    user_id = str(payload.get("user_id") or "")
    if (
        not user_id
        or not payload.get("pass_id")
        or payload.get("phase") not in NEXT_PHASE
    ):
        await _handle_dream_batch_result(entry, rows)
        return
    try:
        context = await _pass_attribution(payload)
    except (UsageActivationUnavailable, ValueError):
        await _fail_unattributed_pass(entry)
        raise
    with restore_cost_context(user_id, context):
        await _handle_dream_batch_result(entry, rows)


async def _pass_attribution(payload: dict[str, Any]) -> TrialCostContext:
    """The usage attribution the pass started under: its row's
    (``store.read_cost_context``), the one source every charge of the pass
    goes by. The batch's payload carries the copy #15208 put there, which
    must agree with it. While the row cannot be read (the store is down),
    the copy stands in for this callback's own work, so a store outage does
    not end the pass; its charges (``batch_costs``) still wait on the row.

    Raises ``UsageActivationUnavailable`` when the payload carries no copy,
    the row holds no attribution (none was kept, or there is no row), or
    the two differ; ``ValueError`` when the copy does not parse or is
    another user's."""
    user_id = str(payload["user_id"])
    carried = payload.get("cost_context")
    if carried is None:
        raise UsageActivationUnavailable("Dream batch has no usage attribution")
    copy = TrialCostContext.model_validate(carried)
    if copy.user_id != user_id:
        raise ValueError("Trial cost attribution belongs to a different user")
    pass_id = str(payload["pass_id"])
    try:
        recorded = await read_cost_context(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its usage attribution; the "
            "callback goes on under its batch's copy",
            exc_info=True,
        )
        return copy
    if recorded is None:
        raise UsageActivationUnavailable("Dream pass has no usage attribution")
    if recorded != copy:
        raise UsageActivationUnavailable(
            "Dream batch attribution differs from its pass's"
        )
    return recorded


async def _fail_unattributed_pass(entry: PendingEntry) -> None:
    """Close a pass whose usage attribution could not be established for its
    batch's result (#15208): errored like any failed pass (its admin job,
    its record with what the landed phases used, the disowned lock, its
    state, the batch stopped at the provider), and charged only under the
    attribution its row carries, if any (``batch_costs``): never under
    another. The lock is the pass's own scope's, read off its input bundle
    as the phase handlers read it; without the bundle there is no token to
    release it with, and it is left to its TTL. The batch is the one this
    result came from: a result the executor gave up waiting for (its
    lifetime timeout) may still be running, and the row stays marked until
    the provider says it has stopped (``cleanup.py``)."""
    bp = BatchPass.from_payload(entry.payload or {})
    try:
        input_bundle = await read_input_bundle(bp.pass_id)
    except Exception:
        logger.exception("Unattributed dream batch pass=%s: no bundle", bp.pass_id)
        input_bundle = None
    if input_bundle is not None:
        bp = bp.model_copy(
            update={
                "user_id": input_bundle.user_id,
                "expert_id": input_bundle.expert_id,
            }
        )
    await fail_pass(bp, UNATTRIBUTED_ERROR, provider_batch_id=entry.provider_batch_id)


async def _handle_dream_batch_result(
    entry: PendingEntry, rows: list[BatchResultRow]
) -> None:
    """One finished phase batch, under its pass's attribution: validate and keep
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
        await end_dead_end(bp, pass_error("batch", "payload_incomplete"))
        return
    if phase not in NEXT_PHASE:
        logger.warning("Dream batch handler unknown phase=%r", phase)
        await end_dead_end(bp, pass_error("batch", "unknown_phase"))
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
            await fail_pass(bp, pass_error("batch", "input_missing"))
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
            await fail_pass(scoped, pass_error("batch", "scope_mismatch"))
            return
        await _handle_phase_result(bp, input_bundle, phase, rows)
    except Exception:
        logger.exception(
            "Dream batch handler crashed for pass=%s phase=%s", bp.pass_id, phase
        )
        await _fail_after_crash(scoped, pass_error(phase, "crashed"))


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
        await fail_pass(bp, pass_error(phase, "no_rows"))
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
        await write_phase_to_state(
            pass_id=bp.pass_id, phase=phase, row=row, model=bp.phase_models.get(phase)
        )
        logger.warning(
            "Dream batch pass=%s phase=%s errored: %s", bp.pass_id, phase, row.error
        )
        await fail_pass(bp, pass_error(phase, "provider_error"))
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
        row.error = f"invalid output shape — {type(exc).__name__}"
        await write_phase_to_state(
            pass_id=bp.pass_id, phase=phase, row=row, model=bp.phase_models.get(phase)
        )
        logger.warning(
            "Dream batch pass=%s phase=%s answered off its schema: %s",
            bp.pass_id,
            phase,
            type(exc).__name__,
        )
        await fail_pass(bp, pass_error(phase, "schema_mismatch"))
        return None
    await write_phase_to_state(
        pass_id=bp.pass_id,
        phase=phase,
        row=row.with_content(json.dumps(payload)),
        model=bp.phase_models.get(phase),
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
        await fail_pass(bp, pass_error(next_phase, "no_api_key"))
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
    except Exception:
        logger.exception(
            "Failed to submit %s phase for pass=%s — marking errored",
            next_phase,
            bp.pass_id,
        )
        await fail_pass(bp, pass_error(next_phase, "submit_failed"))
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
        await fail_pass(bp, pass_error("apply", "import_failed"))
        return
    state = await read_state(bp.pass_id)
    ops = await _terminal_ops(bp, state, input_bundle)
    lease = await _claim_apply(bp, ops) if ops is not None else None
    if ops is None or lease is None:
        return
    try:
        # No ingestion drain here: walk_once awaits this handler serially, so
        # an in-line drain would stall every other user's pending batch
        # (``BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS``). The input bundle is
        # hours old by now; the demotion writes test the recall stamps as the
        # graph holds them when each runs (``demotions.py``).
        apply_stats = await apply_operations(
            MemoryScope.build(bp.user_id, bp.expert_id),
            bp.pass_id,
            ops,
            known_fact_uuids=input_bundle.known_fact_uuids,
            known_episode_uuids=input_bundle.known_episode_uuids,
            fact_scopes=fact_scopes(input_bundle),
            ingestion_drain_timeout=BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS,
            lease=lease,
        )
    except DreamLockLostError:
        logger.exception(
            "apply_operations lost the dream lock for batch pass=%s — marking errored",
            bp.pass_id,
        )
        await fail_pass(bp, pass_error("apply", "lock_lost"))
        return
    except Exception:
        logger.exception(
            "apply_operations crashed for batch pass=%s — marking errored",
            bp.pass_id,
        )
        await fail_pass(bp, pass_error("apply", "crashed"))
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
    lock_token = await lock_token_of(bp.pass_id)
    if await end_batch_pass_if_stopped(bp):
        return None
    gate = await claim_apply_gate(bp.pass_id)
    if gate == "error":
        # We cannot tell first-vs-duplicate apart, and "complete with no
        # writes" would silently drop the dream.
        await fail_pass(bp, pass_error("apply", "gate_unavailable"))
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
        await fail_pass(bp, pass_error("sanitize", "missing_content"))
        return None
    try:
        ops = DreamOperations.model_validate(json.loads(sanitize_row["content"]))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "Dream batch pass=%s sanitize answered off its schema: %s",
            bp.pass_id,
            type(exc).__name__,
        )
        await fail_pass(bp, pass_error("sanitize", "schema_mismatch"))
        return None
    # Enforce the same per-pass operation caps the sync path applies
    # before writing — the model can over-emit past the prompt's limits.
    # The 5%-of-active-facts demotion ceiling needs the original fact
    # count, and the known-fact allowlist filters hallucinated demotion
    # uuids BEFORE the cap slice (else they displace valid demotions).
    return clamp_pass_operations(ops, input_bundle)


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
