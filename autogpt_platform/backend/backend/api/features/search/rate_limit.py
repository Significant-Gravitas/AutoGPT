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

Store search (``/api/store/search``, ``/api/store/agents?search_query=``)
is public, so it is limited per account when signed in and per client address
otherwise (:func:`enforce_public_search_rate_limit`), and the new query embeddings it
requests share one per-minute budget across all clients
(:func:`take_public_search_embedding_slot`).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from datetime import UTC, datetime
from typing import Any, cast

import fastapi
import sentry_sdk
from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.monitoring.instrumentation import record_rate_limit_hit
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# 120 req/minute / user = 2 QPS sustained, generous for the 200 ms-debounced
# search-as-you-type frontend (~5 keystrokes/sec collapse to ~1 call/sec)
# while still catching runaway clients.
GLOBAL_SEARCH_WINDOW_SECONDS = 60
GLOBAL_SEARCH_MAX_REQUESTS = 120

PUBLIC_SEARCH_WINDOW_SECONDS = 60
PUBLIC_SEARCH_EMBEDDINGS_KEY_PREFIX = "store:search:embeddings"
PUBLIC_SEARCH_EMBEDDINGS_ALERT_PERCENT = 80
PUBLIC_SEARCH_EMBEDDINGS_ALERT_TAG = "store_search_embedding_budget"
# One short deadline for the connect and the command: an outage must fall
# through fast instead of holding the request on redis-py's retry ladder.
PUBLIC_SEARCH_REDIS_TIMEOUT_SECONDS = 0.25

_REDIS_FAILURES = (
    RedisError,
    RedisClusterException,
    ConnectionError,
    OSError,
    asyncio.TimeoutError,
    ValueError,
)

# INCR, and set the TTL only on the INCR that opened the window, in one atomic
# step so a new key can never be left without an expiry.
_INCR_OPEN_WINDOW = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


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
        record_rate_limit_hit("/api/search", user_id)
        raise fastapi.HTTPException(
            status_code=429,
            detail=(
                f"Global search rate limit exceeded "
                f"({GLOBAL_SEARCH_MAX_REQUESTS} requests per "
                f"{GLOBAL_SEARCH_WINDOW_SECONDS}s). Try again shortly."
            ),
        )


async def enforce_public_search_rate_limit(client: str, *, user_id: str | None) -> None:
    """Raise HTTP 429 with ``Retry-After`` when the caller has made more store
    searches this minute than its limit: ``public_search_user_requests_per_minute``
    for a signed-in account, ``public_search_requests_per_minute`` for a signed-out
    ``client`` address.

    Fails open on Redis trouble; the embedding budget still bounds the cost.
    """
    if user_id:
        caller = f"user:{user_id}"
        limit = settings.config.public_search_user_requests_per_minute
    else:
        caller = f"ip:{client}"
        limit = settings.config.public_search_requests_per_minute
    now = datetime.now(UTC)
    caller_key = hashlib.sha256(caller.encode()).hexdigest()[:16]
    bucket = int(now.timestamp()) // PUBLIC_SEARCH_WINDOW_SECONDS
    try:
        count = await _count_in_window(f"store:search:rl:{caller_key}:{bucket}")
    except _REDIS_FAILURES as e:
        logger.warning(f"Store search rate-limit check failed open: {e!r}")
        return

    if count > limit:
        record_rate_limit_hit("/api/store/search", caller_key)
        raise fastapi.HTTPException(
            status_code=429,
            detail=(
                f"Store search rate limit exceeded ({limit} requests per "
                f"{PUBLIC_SEARCH_WINDOW_SECONDS}s). Try again shortly."
            ),
            headers={"Retry-After": str(_seconds_left_in_window(now))},
        )


async def take_public_search_embedding_slot() -> bool:
    """Whether store search may request one more new query embedding this
    minute, out of ``public_search_embeddings_per_minute`` shared by all clients.

    Fails closed on Redis trouble: the search still answers, ranked by text.
    Records one Sentry event a minute when the budget is
    ``PUBLIC_SEARCH_EMBEDDINGS_ALERT_PERCENT`` percent used, and one when it is spent.
    A budget of 0 turns embeddings off: no slot, no count, no alert.
    """
    if not public_search_embeddings_enabled():
        return False
    now = datetime.now(UTC)
    bucket = int(now.timestamp()) // PUBLIC_SEARCH_WINDOW_SECONDS
    try:
        count = await _count_in_window(
            f"{PUBLIC_SEARCH_EMBEDDINGS_KEY_PREFIX}:{bucket}"
        )
    except _REDIS_FAILURES as e:
        logger.warning(f"Store search embedding budget unavailable: {e!r}")
        return False

    limit = settings.config.public_search_embeddings_per_minute
    # Each count is reached by exactly one call per window, so each fires once.
    if count == math.ceil(limit * PUBLIC_SEARCH_EMBEDDINGS_ALERT_PERCENT / 100):
        _alert_embedding_budget(
            "nearly_spent",
            f"Store search has used {count} of its {limit} query embeddings "
            "for this minute",
        )
    elif count == limit + 1:
        _alert_embedding_budget(
            "spent",
            f"Store search used its {limit} query embeddings for this minute; "
            "ranking by text alone until the next one",
        )
    return count <= limit


def public_search_embeddings_enabled() -> bool:
    """False when ``public_search_embeddings_per_minute`` is 0: store search then
    always ranks by text alone."""
    return settings.config.public_search_embeddings_per_minute > 0


def _alert_embedding_budget(stage: str, message: str) -> None:
    """Log at ERROR so the Sentry event exists, tagged and fingerprinted per stage
    so an alert rule can select it and each stage stays one issue."""
    tags = {"alert": PUBLIC_SEARCH_EMBEDDINGS_ALERT_TAG, "budget_stage": stage}
    with sentry_sdk.new_scope() as scope:
        for tag, value in tags.items():
            scope.set_tag(tag, value)
        scope.fingerprint = [PUBLIC_SEARCH_EMBEDDINGS_ALERT_TAG, stage]
        logger.error(message, extra={"json_fields": tags})


async def _count_in_window(key: str) -> int:
    """This call's position in the window ``key`` counts."""
    return await asyncio.wait_for(
        _incr_open_window(key), timeout=PUBLIC_SEARCH_REDIS_TIMEOUT_SECONDS
    )


async def _incr_open_window(key: str) -> int:
    redis = await get_redis_async()
    return await cast(
        Any,
        redis.eval(_INCR_OPEN_WINDOW, 1, key, str(PUBLIC_SEARCH_WINDOW_SECONDS)),
    )


def _seconds_left_in_window(now: datetime) -> int:
    return PUBLIC_SEARCH_WINDOW_SECONDS - (
        int(now.timestamp()) % PUBLIC_SEARCH_WINDOW_SECONDS
    )
