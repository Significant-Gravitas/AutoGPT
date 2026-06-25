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

Per-pass state lives in two Redis keys:

  * ``dream:batch:input:{pass_id}`` — the serialized ``DreamInput``
    (so we can rebuild each phase's prompt without re-fetching from
    Postgres / FalkorDB) plus the dream lock's ownership token for the
    compare-and-delete release
  * ``dream:batch:state:{pass_id}`` — accumulated phase outputs +
    per-phase token usage so the apply step has everything it needs
    and the cost log can record all three rows at once

Both are TTL'd to 24h (Anthropic's batch SLA) so a forgotten pass
naturally falls off the radar. Two SETNX gates (7-day TTL) keep the
side effects at-most-once across batch re-dispatch:
``dream:applied:{pass_id}`` for the memory writes and
``dream:batch:costs_logged:{pass_id}`` for billing.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from .batch_submit import (
    PHASE_RESPONSE_MODELS,
    delete_input_bundle,
    read_input_bundle,
    read_lock_token,
    submit_phase,
)
from .billing import record_phase_cost
from .locks import release_dream_lock
from .model_pricing import compute_cost_usd
from .schemas import DreamOperations, DreamOperationsSnapshot, PhaseUsage

if TYPE_CHECKING:
    from backend.executor.batch_executor import PendingEntry
    from backend.util.llm.providers import BatchResultRow

logger = logging.getLogger(__name__)


NAMESPACE = "dream_pass"

DreamPhase = Literal["consolidate", "recombine", "sanitize"]

NEXT_PHASE: dict[DreamPhase, DreamPhase | None] = {
    "consolidate": "recombine",
    "recombine": "sanitize",
    "sanitize": None,  # terminal — apply runs from here
}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _state_key(pass_id: str) -> str:
    return f"dream:batch:state:{pass_id}"


# 24h matches Anthropic's batch SLA; if no phase has landed in that
# window the BatchExecutor has already issued a timeout error via
# ``MAX_BATCH_LIFETIME_SECONDS``.
STATE_TTL_SECONDS = 24 * 60 * 60


async def _read_state(pass_id: str) -> dict[str, dict[str, Any]]:
    """Per-pass accumulator: phase → {content, usage tokens, error}.

    Returns a ``dict[str, ...]`` rather than ``dict[DreamPhase, ...]``
    because Redis hash keys are plain strings and we lose the
    ``Literal`` narrowing the moment we read them back. Callers
    re-narrow at the boundary (e.g. via ``NEXT_PHASE`` lookups) when
    they need the phase ordering.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await redis.hgetall(_state_key(pass_id))  # type: ignore[misc]
    out: dict[str, dict[str, Any]] = {}
    for phase, body in (raw or {}).items():
        if isinstance(phase, bytes):
            phase = phase.decode("utf-8")
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        try:
            out[phase] = json.loads(body)
        except Exception:
            logger.warning("Corrupted state row for pass=%s phase=%s", pass_id, phase)
    return out


async def _write_phase_to_state(
    *, pass_id: str, phase: DreamPhase, row: "BatchResultRow"
) -> None:
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    body = json.dumps(
        {
            "custom_id": row.custom_id,
            "content": row.content,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "cache_read_tokens": row.cache_read_tokens,
            "cache_creation_tokens": row.cache_creation_tokens,
            "error": row.error,
        }
    )
    await redis.hset(_state_key(pass_id), phase, body)  # type: ignore[misc]
    await redis.expire(_state_key(pass_id), STATE_TTL_SECONDS)


async def _delete_state(pass_id: str) -> None:
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.delete(_state_key(pass_id))


def _phase_models_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Per-phase model map persisted by ``submit_phase`` — used to chain
    the next phase and to price each phase with the model it actually
    used. Empty when absent (current code never omits it)."""
    raw = payload.get("phase_models")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


async def _mark_job_errored_best_effort(job_id: str, error: str) -> None:
    """Close the admin job row on a dead-end the normal fail path can't
    reach (malformed payload). Best-effort: status write failures are
    logged, never raised."""
    if not job_id:
        return
    try:
        from .job_status import mark_errored

        await mark_errored(kind="dream_pass", job_id=job_id, error=error)
    except Exception:
        logger.exception("Failed to mark dead-end job %s errored", job_id[:12])


async def _finalize_stuck_duplicate(
    *, user_id: str, pass_id: str, job_id: str, ops: DreamOperations
) -> None:
    """On a duplicate delivery, finalize the job row iff the first
    delivery crashed between apply and ``mark_complete`` and left it
    non-terminal. An already-terminal row is left untouched so the
    first delivery's real apply stats are never overwritten.

    Best-effort: a status read/write failure here must not crash the
    duplicate tail (lock release + cleanup still need to run)."""
    if not job_id:
        return
    try:
        from .job_status import mark_complete, read_status
        from .schemas import DreamPassResult

        existing = await read_status(kind="dream_pass", job_id=job_id)
        if existing is None or existing.state in ("complete", "errored"):
            return
        logger.warning(
            "Duplicate dispatch found job %s stuck in state=%s — "
            "finalizing with the clamped op counts",
            job_id[:12],
            existing.state,
        )
        # The first delivery's per-edge outcomes (and dream session id)
        # died with it, so the counts here are the clamped *attempted*
        # ops — annotate the summary so the admin UI doesn't present
        # them as confirmed apply results.
        note = (
            "[finalized after duplicate delivery — counts reflect attempted "
            "operations; writes landed with the original delivery] "
        )
        await mark_complete(
            kind="dream_pass",
            job_id=job_id,
            result=DreamPassResult(
                user_id=user_id,
                pass_id=pass_id,
                execution_path="anthropic_batch",
                consolidated_count=len(ops.writes),
                proposal_count=len(ops.proposals),
                demotion_count=len(ops.demotions),
                entity_invalidation_count=len(ops.entity_invalidations),
                summary_for_user=note + (ops.summary_for_user or ""),
            ),
        )
    except Exception:
        logger.exception("Failed to finalize stuck duplicate for job %s", job_id[:12])


async def _best_effort_cleanup(pass_id: str) -> None:
    """Delete the per-pass state + input bundle without letting a Redis
    blip propagate. These deletes run AFTER ``mark_complete`` on the
    success/duplicate tails — an exception here would route through the
    crash guard to ``_fail_pass`` and rewrite a completed job to errored.
    Both keys carry 24h TTLs, so a failed delete self-heals."""
    try:
        await _delete_state(pass_id)
        await delete_input_bundle(pass_id)
    except Exception:
        logger.exception(
            "Per-pass cleanup failed for pass=%s — keys will expire via TTL",
            pass_id,
        )


async def _release_lock(user_id: str, pass_id: str) -> None:
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
    if pass_id:
        try:
            token = await read_lock_token(pass_id)
        except Exception:
            logger.exception(
                "Failed to read dream lock token for pass=%s — "
                "leaving the lock to its TTL",
                pass_id,
            )
    await release_dream_lock(user_id, token)


# ---------------------------------------------------------------------------
# Handler entry point
# ---------------------------------------------------------------------------


async def handle_dream_batch_result(
    entry: "PendingEntry", rows: list["BatchResultRow"]
) -> None:
    """BatchExecutor entry — called once per finished phase batch.

    Reads the phase label off ``entry.payload``, validates the
    response shape, persists it to the per-pass accumulator, then
    decides what to do next:

      * If error → mark JobStatus errored, clean up state + input
      * If non-terminal phase → submit next phase batch, leave job
        in ``submitted`` state with updated ``current_phase``
      * If terminal phase (sanitize) → run apply, log costs for all
        three phases, mark JobStatus complete, clean up
    """
    payload = entry.payload or {}
    user_id = str(payload.get("user_id") or "")
    pass_id = str(payload.get("pass_id") or "")
    job_id = str(payload.get("job_id") or "")
    phase_models = _phase_models_from_payload(payload)
    phase = payload.get("phase")

    if not user_id or not pass_id or not phase:
        logger.warning(
            "Dream batch handler missing user_id/pass_id/phase — payload=%s",
            payload,
        )
        # Dead-end payload: close the admin job row (it would otherwise sit
        # queued/submitted until its TTL) and release the disowned lock so
        # the user isn't locked out until the 24h TTL.
        await _mark_job_errored_best_effort(
            job_id, "batch payload missing user_id/pass_id/phase"
        )
        if user_id:
            await _release_lock(user_id, pass_id)
        return

    if phase not in NEXT_PHASE:
        logger.warning("Dream batch handler unknown phase=%r", phase)
        await _mark_job_errored_best_effort(job_id, f"unknown batch phase {phase!r}")
        await _release_lock(user_id, pass_id)
        return

    try:
        await _handle_phase_result(
            rows=rows,
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            phase=phase,
        )
    except Exception:
        # The batch path disowned the per-user dream lock to this callback,
        # and BatchExecutor._dispatch swallows handler exceptions — so without
        # this guard an unexpected error here would strand the user behind the
        # disowned lock until its extended TTL expired and leave the admin
        # JobStatus row stuck. Route the crash through _fail_pass (releases the
        # lock + marks the job errored); if even that fails, release the lock
        # directly so the user is never blocked on a leaked lock.
        logger.exception(
            "Dream batch handler crashed for pass=%s phase=%s", pass_id, phase
        )
        try:
            await _fail_pass(
                user_id=user_id,
                pass_id=pass_id,
                job_id=job_id,
                phase_models=phase_models,
                error=f"{phase}: handler crashed",
            )
        except Exception:
            logger.exception("Dream batch _fail_pass also failed for pass=%s", pass_id)
            try:
                await _release_lock(user_id, pass_id)
            except Exception:
                logger.exception(
                    "Dream batch lock release failed for user=%s", user_id[:12]
                )


async def _handle_phase_result(
    *,
    rows: list["BatchResultRow"],
    user_id: str,
    pass_id: str,
    job_id: str,
    phase_models: dict[str, str],
    phase: DreamPhase,
) -> None:
    """Validate one finished phase batch, then chain to the next phase or
    finalize.

    Split out from ``handle_dream_batch_result`` so the latter can wrap
    this in a single crash guard that always releases the disowned dream
    lock — every early-return below already finalizes via ``_fail_pass``,
    but an *unexpected* raise (Redis blip, apply bug) must not leak the
    lock either.
    """
    if not rows:
        logger.warning("Dream batch handler got empty rows for pass=%s", pass_id)
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"{phase}: provider returned no rows",
        )
        return

    # Single-request-per-batch today; the first (and only) row is the
    # phase result. When we group batches in the future the BatchExecutor
    # will already split by custom_id before calling us.
    row = rows[0]
    if row.error:
        await _write_phase_to_state(pass_id=pass_id, phase=phase, row=row)
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"{phase}: {row.error}",
        )
        return

    # Validate the row's content matches the phase's Pydantic schema
    # BEFORE persisting — corrupted content shouldn't pollute the
    # accumulator for the next phase to read back.
    try:
        PHASE_RESPONSE_MODELS[phase].model_validate(json.loads(row.content))
    except (json.JSONDecodeError, ValidationError) as exc:
        await _write_phase_to_state(pass_id=pass_id, phase=phase, row=row)
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"{phase}: invalid output shape — {type(exc).__name__}",
        )
        return

    await _write_phase_to_state(pass_id=pass_id, phase=phase, row=row)

    next_phase = NEXT_PHASE[phase]
    if next_phase is not None:
        await _chain_next_phase(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            next_phase=next_phase,
        )
        return

    # Terminal phase landed — apply + finalize.
    await _finalize_complete(
        user_id=user_id,
        pass_id=pass_id,
        job_id=job_id,
        phase_models=phase_models,
    )


# ---------------------------------------------------------------------------
# Phase chaining
# ---------------------------------------------------------------------------


async def _chain_next_phase(
    *,
    user_id: str,
    pass_id: str,
    job_id: str,
    phase_models: dict[str, str],
    next_phase: DreamPhase,
) -> None:
    """Submit the next phase in the chain.

    Reads the persisted ``DreamInput`` + accumulated prior phase
    outputs from Redis, builds the next phase's prompt, fires another
    batch submission. On any failure to submit, marks the JobStatus
    errored — silent submission failures are unrecoverable.
    """
    input_bundle = await read_input_bundle(pass_id)
    if input_bundle is None:
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"{next_phase}: DreamInput missing from Redis (TTL expired?)",
        )
        return

    state = await _read_state(pass_id)
    consolidated_json = _content_for(state, "consolidate")
    recombined_json = _content_for(state, "recombine")

    api_key = _anthropic_api_key()
    if api_key is None:
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"{next_phase}: no Anthropic API key configured",
        )
        return

    try:
        submission = await submit_phase(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase=next_phase,
            phase_models=phase_models,
            api_key=api_key,
            input_bundle=input_bundle,
            consolidated_json=consolidated_json,
            recombined_json=recombined_json,
        )
    except Exception as exc:
        logger.exception(
            "Failed to submit %s phase for pass=%s — marking errored",
            next_phase,
            pass_id,
        )
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"{next_phase}: submit failed: {type(exc).__name__}: {exc}",
        )
        return

    if job_id:
        try:
            from .job_status import update_status_phase

            await update_status_phase(
                kind="dream_pass",
                job_id=job_id,
                state="submitted",
                current_phase=next_phase,
                batch_id=submission.provider_batch_id,
            )
        except Exception:
            logger.exception(
                "Failed to update status for next phase=%s pass=%s",
                next_phase,
                pass_id,
            )


def _content_for(state: dict[str, dict[str, Any]], phase: str) -> str | None:
    row = state.get(phase)
    if row is None:
        return None
    content = row.get("content")
    return content if isinstance(content, str) else None


# ---------------------------------------------------------------------------
# Terminal handlers
# ---------------------------------------------------------------------------


_APPLIED_GATE_PREFIX = "dream:applied"
# 7 days — same window as the costs_logged gate; no realistic
# BatchExecutor re-dispatch (poll backoff caps at 5 min, max lifetime
# 24h) can outlive it.
_APPLIED_GATE_TTL_SECONDS = 7 * 24 * 60 * 60


async def _claim_apply_gate(pass_id: str) -> Literal["claimed", "duplicate", "error"]:
    """Atomically claim the per-pass apply gate.

    Returns ``"claimed"`` when this delivery is the first to run
    ``apply_operations`` for the pass, ``"duplicate"`` on a re-dispatched
    delivery whose writes already landed, and ``"error"`` when Redis is
    unavailable and we cannot tell which of the two we are.

    Mirrors ``_claim_costs_logged_gate``: if the BatchExecutor crashes
    between dispatch and ``remove_pending``, the next poll re-dispatches
    the same finished batch — and ``apply_operations`` writes every
    consolidated fact and proposal to the user's graph as fresh episodes,
    so re-running it duplicates the user's memories. The three states must
    stay distinct: treating a Redis brown-out as a duplicate would mark
    the job complete with zero writes, silently dropping the dream.
    """
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        claimed = await redis.set(
            f"{_APPLIED_GATE_PREFIX}:{pass_id}",
            "1",
            nx=True,
            ex=_APPLIED_GATE_TTL_SECONDS,
        )
        return "claimed" if claimed else "duplicate"
    except Exception:
        logger.exception(
            "Failed to claim apply gate for pass=%s — failing pass",
            pass_id,
        )
        return "error"


async def _finalize_complete(
    *, user_id: str, pass_id: str, job_id: str, phase_models: dict[str, str]
) -> None:
    """Sanitize phase has landed. Run apply + cost log + complete."""
    try:
        from .apply import apply_operations
        from .orchestrator import _clamp_operations
    except Exception:
        logger.exception("Failed to import dream apply for pass=%s", pass_id)
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error="apply: import failed",
        )
        return

    state = await _read_state(pass_id)
    sanitize_row = state.get("sanitize")
    if sanitize_row is None or not sanitize_row.get("content"):
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error="sanitize: missing terminal phase content",
        )
        return

    try:
        ops = DreamOperations.model_validate(json.loads(sanitize_row["content"]))
    except (json.JSONDecodeError, ValidationError) as exc:
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"sanitize: shape validation failed: {type(exc).__name__}",
        )
        return

    # Enforce the same per-pass operation caps the sync path applies
    # before writing — the model can over-emit past the prompt's limits.
    # The 5%-of-active-facts demotion ceiling needs the original fact
    # count; the persisted input bundle still exists here (deleted only
    # after apply below). A missing bundle (-1) falls back to the
    # absolute demotion cap rather than zeroing all demotions.
    input_bundle = await read_input_bundle(pass_id)
    active_fact_count = len(input_bundle.facts) if input_bundle is not None else -1
    # Pass the known-fact allowlist so hallucinated demotion uuids are
    # filtered BEFORE the cap slice — otherwise they consume cap slots
    # and displace valid demotions (cap can floor at 1 on small graphs).
    ops = _clamp_operations(
        ops,
        active_fact_count,
        known_fact_uuids=(
            input_bundle.known_fact_uuids if input_bundle is not None else None
        ),
    )

    # Batch results can re-dispatch (executor crash between dispatch and
    # ``remove_pending``). Billing below is gated; the memory mutation must
    # be too. A duplicate delivery skips apply AND mark_complete — the first
    # delivery already wrote the real stats, and overwriting them with empty
    # ones would zero the admin-visible counts. A gate error fails the pass:
    # we cannot tell first-vs-duplicate apart, and "complete with no writes"
    # would silently drop the dream.
    gate = await _claim_apply_gate(pass_id)
    if gate == "error":
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error="apply: gate unavailable (redis) — cannot guarantee at-most-once",
        )
        return
    if gate == "duplicate":
        logger.info(
            "Duplicate dispatch for pass=%s — operations already applied; "
            "preserving the first delivery's job result",
            pass_id,
        )
        # Normally the first delivery wrote the terminal status. If it
        # crashed between apply and mark_complete, the row is stuck in
        # 'submitted' — finalize it here WITHOUT clobbering an existing
        # terminal result (the apply counts are this pass's clamped ops;
        # the writes themselves landed with the first delivery).
        await _finalize_stuck_duplicate(
            user_id=user_id, pass_id=pass_id, job_id=job_id, ops=ops
        )
        await _log_all_phase_costs(
            user_id=user_id, pass_id=pass_id, state=state, phase_models=phase_models
        )
        await _release_lock(user_id, pass_id)
        await _best_effort_cleanup(pass_id)
        return

    apply_stats: dict[str, int | str | DreamOperationsSnapshot] = {}
    try:
        # Thread the demotion allowlist from the bundle already in memory —
        # letting apply re-read it would do a second Redis GET + full JSON
        # deserialize and could fail open if the bundle's TTL lapses between
        # the two reads. None (bundle expired) keeps apply's documented
        # fail-open fallback.
        apply_stats = await apply_operations(
            user_id,
            pass_id,
            ops,
            known_fact_uuids=(
                input_bundle.known_fact_uuids if input_bundle is not None else None
            ),
        )
    except Exception as exc:
        logger.exception(
            "apply_operations crashed for batch pass=%s — marking errored",
            pass_id,
        )
        await _fail_pass(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            phase_models=phase_models,
            error=f"apply: {type(exc).__name__}: {exc}",
        )
        return

    # Per-phase usage log on the success path. Failure paths record the
    # same usage via ``_fail_pass`` (we incurred those provider tokens
    # regardless); the Redis dedup gate inside ``_log_all_phase_costs``
    # keeps it at-most-once across both paths and any batch re-dispatch.
    await _log_all_phase_costs(
        user_id=user_id, pass_id=pass_id, state=state, phase_models=phase_models
    )

    if job_id:
        try:
            from .job_status import mark_complete
            from .schemas import DreamPassResult

            raw_snapshot = apply_stats.get("snapshot")
            snapshot: DreamOperationsSnapshot | None = None
            if isinstance(raw_snapshot, DreamOperationsSnapshot):
                snapshot = raw_snapshot
            elif isinstance(raw_snapshot, dict):
                snapshot = DreamOperationsSnapshot.model_validate(raw_snapshot)

            # ``apply_stats`` values are typed as a union that includes
            # ``DreamOperationsSnapshot``; narrow each count to a plain
            # ``int`` (with a 0 default) before threading it into the
            # Pydantic result so pyright doesn't flag the int() cast.
            def _count(key: str) -> int:
                value = apply_stats.get(key)
                if isinstance(value, (int, str)):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return 0
                return 0

            raw_session_id = apply_stats.get("session_id")
            session_id = raw_session_id if isinstance(raw_session_id, str) else None

            pass_result = DreamPassResult(
                user_id=user_id,
                pass_id=pass_id,
                execution_path="anthropic_batch",
                consolidated_count=_count("consolidated_count"),
                proposal_count=_count("proposal_count"),
                demotion_count=_count("demotion_count"),
                entity_invalidation_count=_count("entity_invalidation_count"),
                dream_session_id=session_id,
                operations=snapshot,
                # Carry the user-facing narrative like the sync path does —
                # without it the Memory Visualizer renders a blank summary for
                # batch-completed dreams even though the session message exists.
                summary_for_user=ops.summary_for_user,
            )
            await mark_complete(kind="dream_pass", job_id=job_id, result=pass_result)
        except Exception:
            logger.exception("Failed to mark dream pass job %s complete", job_id)

    # The batch path disowned the dream lock to this callback; release it now
    # that the pass has terminated so the next dream for this user can run.
    await _release_lock(user_id, pass_id)
    await _best_effort_cleanup(pass_id)


async def _fail_pass(
    *,
    user_id: str,
    pass_id: str,
    job_id: str,
    phase_models: dict[str, str],
    error: str,
) -> None:
    """Mark JobStatus errored, record usage for any phases that already
    landed, then clean up per-pass state.

    We incurred the provider tokens for completed phases regardless of
    whether the whole pass landed, so they're recorded against the
    user's usage — matching the sync path and the documented contract in
    ``dream/billing.py``. The idempotency gate inside
    ``_log_all_phase_costs`` keeps this at-most-once even if the batch
    re-dispatches.
    """
    logger.warning("Dream batch pass=%s failed: %s", pass_id, error)
    if job_id:
        try:
            from .job_status import mark_errored

            await mark_errored(kind="dream_pass", job_id=job_id, error=error)
        except Exception:
            logger.exception("Failed to mark dream pass job %s errored", job_id)
    state = await _read_state(pass_id)
    if state:
        await _log_all_phase_costs(
            user_id=user_id,
            pass_id=pass_id,
            state=state,
            phase_models=phase_models,
        )
    # Release the dream lock the batch path disowned to this callback.
    await _release_lock(user_id, pass_id)
    await _best_effort_cleanup(pass_id)


# ---------------------------------------------------------------------------
# Cost logging
# ---------------------------------------------------------------------------


_COSTS_LOGGED_PREFIX = "dream:batch:costs_logged"
# 7 days — long enough that no realistic BatchExecutor re-dispatch
# (poll backoff caps at 5 min, max lifetime 24h) can slip through and
# bill twice. Matches the spirit of the Stripe-reconcile gate's TTL.
_COSTS_LOGGED_TTL_SECONDS = 7 * 24 * 60 * 60


async def _claim_costs_logged_gate(pass_id: str) -> bool:
    """Atomically claim the per-pass cost-charge gate. Returns True
    when this caller won the race (first time costs_logged is set);
    False when a prior caller already charged this pass.

    Modelled on ``rate_limit._maybe_reconcile_stripe_tier`` — Redis
    SETNX with a long TTL is the established convention for "do this
    side-effect at most once per identifier" in this codebase. The
    dedup lives at the dream-batch boundary, not inside
    ``record_cost_usage`` itself (chat legitimately charges every turn).
    """
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        return bool(
            await redis.set(
                f"{_COSTS_LOGGED_PREFIX}:{pass_id}",
                "1",
                nx=True,
                ex=_COSTS_LOGGED_TTL_SECONDS,
            )
        )
    except Exception:
        # Fail closed: if we can't claim the gate, do not charge.
        # Better to under-bill on a Redis brown-out than risk
        # double-billing under retry pressure.
        logger.exception(
            "Failed to claim costs_logged gate for pass=%s — skipping charge",
            pass_id,
        )
        return False


async def _log_all_phase_costs(
    *,
    user_id: str,
    pass_id: str,
    state: dict[str, dict[str, Any]],
    phase_models: dict[str, str],
) -> None:
    """One PlatformCostLog row per phase, tagged ``anthropic_batch``.

    Idempotent via a Redis SETNX gate keyed on ``pass_id``: if the
    BatchExecutor crashes between charging and removing the pending
    batch entry and the next poll re-dispatches the same batch, the
    gate prevents the second charge. The gate is set BEFORE the loop
    so a partial failure mid-loop still leaves the user charged for
    whatever phases landed (matches the documented "partial pass
    charges for completed phases" semantic in ``dream/billing.py``).

    Cost is computed via ``dream/model_pricing.compute_cost_usd`` —
    the dream rate card — so the batch path uses the same native-
    Anthropic token convention (additive cache buckets, not subtracted)
    as the sync path. The ``execution_path="anthropic_batch"`` arg
    applies the 50% batch discount there.

    No-ops on per-phase failure — apply already wrote the user-facing
    memory operations; a cost-log blip shouldn't take that down.
    """
    if not await _claim_costs_logged_gate(pass_id):
        logger.info(
            "Skipping batch cost log for pass=%s — already charged",
            pass_id,
        )
        return

    for phase, row in state.items():
        try:
            if row.get("error"):
                # Phase errored — don't record usage for a phase that
                # didn't complete; downstream phases never ran either.
                continue
            phase_model = phase_models.get(phase)
            if not phase_model:
                logger.warning(
                    "No model recorded for pass=%s phase=%s — skipping cost log",
                    pass_id,
                    phase,
                )
                continue
            input_tokens = int(row.get("input_tokens") or 0)
            output_tokens = int(row.get("output_tokens") or 0)
            cache_read_tokens = int(row.get("cache_read_tokens") or 0)
            cache_creation_tokens = int(row.get("cache_creation_tokens") or 0)
            cost_usd = compute_cost_usd(
                model=phase_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                execution_path="anthropic_batch",
            )
            usage = PhaseUsage(
                phase=phase,  # type: ignore[arg-type]
                model=phase_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cost_usd=cost_usd,
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=usage,
                execution_path="anthropic_batch",
            )
        except Exception:
            logger.exception(
                "Failed to log batch cost for pass=%s phase=%s", pass_id, phase
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_api_key() -> str | None:
    """Look up the Anthropic key from the copilot config first, then
    fall back to the shared settings key.

    Mirrors what ``backend/executor/batch_executor.py::_default_api_key_for``
    does — callbacks live in their own subprocess so they need to
    re-resolve the key on their own.
    """
    try:
        from backend.copilot.config import ChatConfig

        cfg = ChatConfig()
        if cfg.direct_anthropic_api_key:
            return cfg.direct_anthropic_api_key
    except Exception:
        logger.debug("ChatConfig unavailable during dream batch callback")
    try:
        from backend.util.settings import Settings

        key = Settings().secrets.anthropic_api_key
        return key or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Registration (runs at module import)
# ---------------------------------------------------------------------------


def _register() -> None:
    try:
        from backend.executor.batch_executor import register_handler

        register_handler(NAMESPACE, handle_dream_batch_result)
    except Exception:
        logger.exception(
            "Failed to register dream batch handler — "
            "batch results will not dispatch"
        )


_register()
