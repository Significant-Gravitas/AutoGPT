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

Both are TTL'd to an hour past the pass's lease (``INPUT_TTL_SECONDS``), so
the reaper can still charge and delete them and a forgotten pass naturally
falls off the radar. SETNX gates (7-day TTL) keep the side effects
at-most-once should a finished batch be delivered twice or a cleanup run
again: ``dream:applied:{pass_id}`` for the memory writes, and for billing
one ``dream:batch:charged:{pass_id}:{phase}`` per landed phase, claimed right
before that phase is charged. ``dream:batch:costs_logged:{pass_id}`` is the
whole-pass billing gate of earlier builds; this build claims it too, with a
value of its own (``claim_per_phase_charging``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Literal, cast

from .batch_submit import INPUT_TTL_SECONDS, delete_input_bundle
from .schemas import DreamPhase

if TYPE_CHECKING:
    from backend.util.llm.providers import BatchResultRow

logger = logging.getLogger(__name__)

# The same as the input bundle's, see ``batch_submit.INPUT_TTL_SECONDS``.
STATE_TTL_SECONDS = INPUT_TTL_SECONDS

_APPLIED_GATE_PREFIX = "dream:applied"
# 7 days — same window as the costs_logged gate; no realistic
# BatchExecutor re-dispatch (poll backoff caps at 5 min, max lifetime
# 24h) can outlive it.
_APPLIED_GATE_TTL_SECONDS = 7 * 24 * 60 * 60

# The pass-level billing key. Earlier builds claim it with "1" and then
# charge every landed phase; this build claims it with ``_PER_PHASE`` and
# charges each phase under its own claim.
_COSTS_LOGGED_PREFIX = "dream:batch:costs_logged"
_PER_PHASE = "per_phase"
_PHASE_CHARGED_PREFIX = "dream:batch:charged"
# 7 days — long enough that no realistic BatchExecutor re-dispatch
# (poll backoff caps at 5 min, max lifetime 24h) or reaper retry can slip
# through and bill twice. Matches the spirit of the Stripe-reconcile gate's TTL.
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


async def read_state_or_none(pass_id: str) -> dict[str, dict[str, Any]] | None:
    """The pass's state, or ``None`` when it cannot be read, for a caller
    that only reports on the pass (the usage its record carries) and must
    go on without it."""
    try:
        return await read_state(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its batch state for the record",
            exc_info=True,
        )
        return None


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


async def best_effort_cleanup(pass_id: str) -> bool:
    """Delete the per-pass state + input bundle, and say whether both went,
    without letting a Redis blip propagate. These deletes run AFTER
    ``mark_complete`` on the success/duplicate tails — an exception here
    would route through the crash guard to ``fail_pass`` and rewrite a
    completed job to errored. A failed delete is logged and reported, so
    the cleanup it belongs to stays marked for the reaper; both keys carry
    TTLs besides."""
    try:
        await delete_state(pass_id)
        await delete_input_bundle(pass_id)
    except Exception:
        logger.exception(
            "Per-pass cleanup failed for pass=%s — left for the reaper",
            pass_id,
        )
        return False
    return True


async def claim_apply_gate(pass_id: str) -> Literal["claimed", "duplicate", "error"]:
    """Atomically claim the per-pass apply gate.

    Returns ``"claimed"`` when this delivery is the first to run
    ``apply_operations`` for the pass, ``"duplicate"`` when an earlier
    delivery already claimed the gate, and ``"error"`` when Redis is
    unavailable and we cannot tell which of the two we are. A duplicate
    proves only that earlier claim, not that its writes landed: a delivery
    that died between its claim and the end of apply leaves the gate
    claimed and its writes partial.

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


async def claim_per_phase_charging(pass_id: str) -> bool:
    """Whether the pass is charged phase by phase (``claim_phase_charge``):
    ``True`` once the pass-level key holds this build's value, set now or by
    an earlier call; ``False`` when it holds an earlier build's, which
    charged the pass whole, so nothing more is charged. Raises when Redis
    does not answer.

    Earlier builds claim that key before charging every phase, so claiming
    it here too keeps one pass from being charged both ways while two builds
    run side by side."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    key = f"{_COSTS_LOGGED_PREFIX}:{pass_id}"
    if await redis.set(key, _PER_PHASE, nx=True, ex=_COSTS_LOGGED_TTL_SECONDS):
        return True
    held = await redis.get(key)
    return (held.decode() if isinstance(held, bytes) else held) == _PER_PHASE


async def claim_phase_charge(pass_id: str, phase: str) -> bool:
    """Claim the charge of one landed phase of the pass: ``True`` for the
    first caller, which then charges it, ``False`` once it is claimed.
    Raises when Redis does not answer, and a claim that landed all the same
    leaves the phase uncharged: never charged twice.

    Modelled on ``rate_limit._maybe_reconcile_stripe_tier`` — Redis SETNX
    with a long TTL is the established convention for "do this side-effect
    at most once per identifier" in this codebase. The dedup lives at the
    dream-batch boundary, not inside ``record_cost_usage`` itself (chat
    legitimately charges every turn)."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    return bool(
        await redis.set(
            f"{_PHASE_CHARGED_PREFIX}:{pass_id}:{phase}",
            "1",
            nx=True,
            ex=_COSTS_LOGGED_TTL_SECONDS,
        )
    )
