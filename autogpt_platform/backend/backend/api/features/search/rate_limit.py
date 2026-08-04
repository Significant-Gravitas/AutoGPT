"""Per-user QPS rate limit for ``/api/search/global``.

Search-as-you-type calls fan out to a paid OpenAI embedding for every
non-empty query (see ``unified_hybrid_search`` → ``embed_query``). The
200 ms frontend debounce keeps normal typing well under any reasonable
cap, so this limiter exists to put a hard ceiling on key-held clients
or scripts that bypass the debounce and would otherwise burn embedding
spend without backpressure.

Fixed-window counter in Redis (INCR + EXPIRE on first hit). Approximate
by design — under high concurrency two calls may both pass against the
same snapshot, same trade-off as :mod:`backend.copilot.rate_limit`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import fastapi

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# 120 req/minute / user = 2 QPS sustained, generous for the 200 ms-debounced
# search-as-you-type frontend (~5 keystrokes/sec collapse to ~1 call/sec)
# while still catching runaway clients.
GLOBAL_SEARCH_WINDOW_SECONDS = 60
GLOBAL_SEARCH_MAX_REQUESTS = 120


def _window_key(user_id: str, *, now: datetime) -> str:
    """Per-user fixed-window key. Bucket aligned to ``GLOBAL_SEARCH_WINDOW_SECONDS``
    so each user has at most one active counter per window."""
    bucket = int(now.timestamp()) // GLOBAL_SEARCH_WINDOW_SECONDS
    return f"search:global:rl:{user_id}:{bucket}"


async def enforce_global_search_rate_limit(user_id: str) -> None:
    """Raise HTTP 429 when ``user_id`` exceeds the per-window cap.

    On Redis brown-out we fail *open* (log and let the call through):
    the worst case is one user briefly burning a few extra embedding
    calls, which is far less bad than a global Redis blip blocking
    search for every user.
    """
    now = datetime.now(UTC)
    key = _window_key(user_id, now=now)
    try:
        redis = await get_redis_async()
        # Atomic create-with-TTL then INCR. The previous ``INCR`` +
        # conditional ``EXPIRE`` was racy: if EXPIRE failed on the
        # first-hit path (e.g. transient network blip — the exact
        # "Redis brown-out" the except below catches) the key stuck
        # around without a TTL until Redis evicted it. ``SET NX EX``
        # makes the TTL part of the same write that creates the key;
        # subsequent INCRs preserve it.
        await redis.set(key, 0, ex=GLOBAL_SEARCH_WINDOW_SECONDS, nx=True)
        count = await redis.incr(key)
    except Exception as e:
        logger.warning(
            "Global-search rate-limit check failed open for user %s: %s",
            user_id,
            e,
        )
        return

    if count > GLOBAL_SEARCH_MAX_REQUESTS:
        raise fastapi.HTTPException(
            status_code=429,
            detail=(
                f"Global search rate limit exceeded "
                f"({GLOBAL_SEARCH_MAX_REQUESTS} requests per "
                f"{GLOBAL_SEARCH_WINDOW_SECONDS}s). Try again shortly."
            ),
        )
