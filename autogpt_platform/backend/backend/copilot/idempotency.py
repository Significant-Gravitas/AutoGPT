"""Per-session idempotency-key dedup for ``POST /stream``.

Each user-initiated send from the frontend carries a freshly generated
``idempotency_key`` (UUID).  Frontend / network / RMQ-redelivery retries
of the **same** logical send reuse the key, so the backend can spot them
and avoid persisting a duplicate user row + spawning a parallel turn.

The key is intentionally per-CLICK on the frontend (not per-content):
two distinct user clicks of identical text 5 minutes apart get two
different keys and BOTH go through — that's the user's intent.  Refresh
+ retype is caught upstream by the persisted ``lastSubmittedMessageText``
in the Zustand store; this module is the defence-in-depth layer for the
cases the frontend can't see (RMQ redelivery, browser/CDN retry).
"""

import logging

from backend.data.redis_client import get_redis, get_redis_async

logger = logging.getLogger(__name__)

_IDEMPOTENCY_KEY_PREFIX = "idempotency:stream"
# 30 minutes — comfortably longer than any realistic single-turn duration
# (the longest current turns top out ~10 min) but short enough that the
# Redis key set stays bounded.
_IDEMPOTENCY_TTL_SECONDS = 30 * 60


def _redis_key(session_id: str, idempotency_key: str) -> str:
    return f"{_IDEMPOTENCY_KEY_PREFIX}:{session_id}:{idempotency_key}"


async def claim_stream_idempotency_key(
    session_id: str,
    idempotency_key: str,
    *,
    turn_id: str,
) -> str | None:
    """Atomically claim *idempotency_key* for *session_id* against this *turn_id*.

    Returns ``None`` when the claim succeeded — caller proceeds with the
    new turn.  Returns the **existing** ``turn_id`` recorded under the
    key when it had already been claimed — caller should subscribe to
    that turn's SSE stream (via ``stream_registry``) instead of starting
    a parallel turn.

    Fail-open on Redis errors: returns ``None`` so the request goes
    through.  An idempotency-dedup outage MUST NOT take the chat down.
    """
    try:
        redis = await get_redis_async()
        key = _redis_key(session_id, idempotency_key)
        # SET NX EX is atomic — exactly one caller wins.
        claimed = await redis.set(key, turn_id, ex=_IDEMPOTENCY_TTL_SECONDS, nx=True)
        if claimed:
            return None
        existing = await redis.get(key)
        if existing is None:
            # The key existed at SET-NX time but expired before our GET.
            # Treat as a successful claim — the original turn is gone, a
            # fresh one is appropriate.
            return None
        return existing.decode() if isinstance(existing, bytes) else existing
    except Exception:
        logger.warning(
            "Idempotency claim failed for session=%s key=%s — failing open",
            session_id,
            idempotency_key,
            exc_info=True,
        )
        return None


async def get_claimed_turn_id(
    session_id: str,
    idempotency_key: str,
) -> str | None:
    """Read-only lookup; returns the turn_id recorded under the key, or
    ``None`` when no claim exists / Redis is unavailable.

    Used by the executor to detect RMQ-redelivered tasks: if the key is
    bound to a turn_id different from the current task's, the original
    processing is in flight or completed, and the redelivered task
    should be a no-op.
    """
    try:
        redis = await get_redis_async()
        existing = await redis.get(_redis_key(session_id, idempotency_key))
        if existing is None:
            return None
        return existing.decode() if isinstance(existing, bytes) else existing
    except Exception:
        logger.warning(
            "Idempotency lookup failed for session=%s key=%s — failing open",
            session_id,
            idempotency_key,
            exc_info=True,
        )
        return None


def sync_get_claimed_turn_id(
    session_id: str,
    idempotency_key: str,
) -> str | None:
    """Synchronous twin of :func:`get_claimed_turn_id`.

    The CoPilot RMQ consumer callback runs on pika's blocking IO thread
    and has no event loop handy, so it uses this sync variant to spot
    redelivered tasks before we burn a thread-pool slot on a duplicate.
    Same fail-open contract — Redis hiccup never wedges the executor.
    """
    try:
        # ``redis-py``'s typing returns ``ResponseT`` (a union over sync /
        # async return types).  We're calling on a sync client, so the
        # value is always ``bytes | str | None`` at runtime.
        existing = get_redis().get(_redis_key(session_id, idempotency_key))
        if existing is None:
            return None
        if isinstance(existing, bytes):
            return existing.decode()
        if isinstance(existing, str):
            return existing
        return None
    except Exception:
        logger.warning(
            "Idempotency lookup failed for session=%s key=%s — failing open",
            session_id,
            idempotency_key,
            exc_info=True,
        )
        return None
