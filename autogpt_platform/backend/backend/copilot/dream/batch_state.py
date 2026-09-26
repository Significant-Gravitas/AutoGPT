"""The Redis state a batch dream pass keeps between its callbacks, and the
gates that keep its side effects at-most-once.

Per-pass state lives in two Redis keys:

  * ``dream:batch:input:{pass_id}`` — the serialized ``DreamInput`` (so each
    phase's prompt is rebuilt without re-fetching from Postgres / FalkorDB)
    plus the dream lock's ownership token for the compare-and-delete release;
    written and read by ``batch_submit.py``
  * ``dream:batch:state:{pass_id}`` — accumulated phase outputs + per-phase
    token usage, so the apply step has everything it needs and the cost log
    can record all three rows at once

Both are TTL'd to 24h (Anthropic's batch SLA) so a forgotten pass naturally
falls off the radar. Two SETNX gates (7-day TTL) keep the side effects
at-most-once should a finished batch be delivered twice:
``dream:applied:{pass_id}`` for the memory writes and
``dream:batch:costs_logged:{pass_id}`` for billing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Literal, cast

from .batch_submit import delete_input_bundle
from .schemas import DreamPhase

if TYPE_CHECKING:
    from backend.util.llm.providers import BatchResultRow

logger = logging.getLogger(__name__)

# 24h matches Anthropic's batch SLA; if no phase has landed in that
# window the BatchExecutor has already issued a timeout error via
# ``MAX_BATCH_LIFETIME_SECONDS``.
STATE_TTL_SECONDS = 24 * 60 * 60

_APPLIED_GATE_PREFIX = "dream:applied"
# 7 days — same window as the costs_logged gate; no realistic
# BatchExecutor re-dispatch (poll backoff caps at 5 min, max lifetime
# 24h) can outlive it.
_APPLIED_GATE_TTL_SECONDS = 7 * 24 * 60 * 60

_COSTS_LOGGED_PREFIX = "dream:batch:costs_logged"
# 7 days — long enough that no realistic BatchExecutor re-dispatch
# (poll backoff caps at 5 min, max lifetime 24h) can slip through and
# bill twice. Matches the spirit of the Stripe-reconcile gate's TTL.
_COSTS_LOGGED_TTL_SECONDS = 7 * 24 * 60 * 60


def state_key(pass_id: str) -> str:
    return f"dream:batch:state:{pass_id}"


async def read_state(pass_id: str) -> dict[str, dict[str, Any]]:
    """Per-pass accumulator: phase → {content, usage tokens, error}.

    Returns a ``dict[str, ...]`` rather than ``dict[DreamPhase, ...]``
    because Redis hash keys are plain strings and we lose the
    ``Literal`` narrowing the moment we read them back. Callers
    re-narrow at the boundary (e.g. via ``NEXT_PHASE`` lookups) when
    they need the phase ordering.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await cast("Awaitable[dict[Any, Any]]", redis.hgetall(state_key(pass_id)))
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


async def write_phase_to_state(
    *, pass_id: str, phase: DreamPhase, row: BatchResultRow
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
    await cast("Awaitable[int]", redis.hset(state_key(pass_id), phase, body))
    await redis.expire(state_key(pass_id), STATE_TTL_SECONDS)


async def delete_state(pass_id: str) -> None:
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.delete(state_key(pass_id))


def content_for(state: dict[str, dict[str, Any]], phase: str) -> str | None:
    row = state.get(phase)
    if row is None:
        return None
    content = row.get("content")
    return content if isinstance(content, str) else None


async def best_effort_cleanup(pass_id: str) -> None:
    """Delete the per-pass state + input bundle without letting a Redis
    blip propagate. These deletes run AFTER ``mark_complete`` on the
    success/duplicate tails — an exception here would route through the
    crash guard to ``fail_pass`` and rewrite a completed job to errored.
    Both keys carry 24h TTLs, so a failed delete self-heals."""
    try:
        await delete_state(pass_id)
        await delete_input_bundle(pass_id)
    except Exception:
        logger.exception(
            "Per-pass cleanup failed for pass=%s — keys will expire via TTL",
            pass_id,
        )


async def claim_apply_gate(pass_id: str) -> Literal["claimed", "duplicate", "error"]:
    """Atomically claim the per-pass apply gate.

    Returns ``"claimed"`` when this delivery is the first to run
    ``apply_operations`` for the pass, ``"duplicate"`` on a repeated
    delivery whose writes already landed, and ``"error"`` when Redis is
    unavailable and we cannot tell which of the two we are.

    The BatchExecutor claims each finished batch atomically before it
    dispatches it (``BatchExecutor._claim_dispatch``), so the same result
    comes back only through a re-enqueued pending entry. The gate keeps the
    memory mutation at-most-once even then: ``apply_operations`` writes
    every consolidated fact and proposal to the user's graph as fresh
    episodes, so running it twice duplicates the user's memories. The three
    states must stay distinct: treating a Redis brown-out as a duplicate
    would mark the job complete with zero writes, silently dropping the
    dream.
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


async def claim_costs_logged_gate(pass_id: str) -> bool:
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
