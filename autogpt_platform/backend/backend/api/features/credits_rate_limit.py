"""Per-user rate limit for ``GET /api/credits/subscription``.

The subscription-status endpoint fans out to uncached Stripe reads per request
and is fetched on every load of the billing page, so a scripted client can
drive a lot of upstream traffic from one account. A per-user cap puts a hard
ceiling on that without affecting normal use: the frontend shares one React
Query entry with a 60s ``staleTime``, so a real user stays well under the cap.

Atomic fixed-window counter in Redis (Lua ``INCR`` + first-hit ``EXPIRE``),
keyed per ``user_id``. A single key per check keeps it correct on the Redis
cluster.

Availability: this puts Redis in front of a billing read, so the check is
strictly best-effort. It is bounded by a short deadline and **fails open** on
any Redis trouble — being unable to prove a user is under their cap must never
cost every user their billing status.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

import fastapi
from autogpt_libs.auth import get_user_id
from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.monitoring.instrumentation import record_rate_limit_hit

logger = logging.getLogger(__name__)

# 60 requests / minute / user — roughly 4x the heaviest realistic legitimate
# burst (a multi-tab simultaneous hard-reload is ~15/min); the shared React
# Query entry plus 60s staleTime keep steady-state use near ~1/min per tab.
SUBSCRIPTION_STATUS_WINDOW_SECONDS = 60
SUBSCRIPTION_STATUS_MAX_REQUESTS = 60

# Hard deadline on the whole Redis interaction. Without it "fail open" is not
# "fail fast": a cold client runs redis-py's own connect retry ladder (and each
# command its per-command retry), so during an outage a request would park an
# ASGI worker slot for minutes instead of falling through. The limiter is
# best-effort, so it gets one short budget and is skipped if Redis can't answer
# inside it.
SUBSCRIPTION_STATUS_REDIS_TIMEOUT_SECONDS = 0.25

# Atomic fixed-window counter: INCR the key, and set the TTL only on the INCR
# that opened the window (count == 1). Doing both in one server-side script
# means a freshly-created key always gets its expiry — the key can never linger
# without a TTL if it happened to be recreated by INCR, and the window stays
# fixed (the TTL is not refreshed on later hits).
_INCR_OPEN_WINDOW = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


async def _incr_window(key: str) -> int:
    """Run the atomic counter, returning the request's position in the window.

    Split out so the connect and the command share one deadline in the caller.
    Lua ARGV values are strings; ``EXPIRE`` coerces ``"60"`` back to an int. The
    cast mirrors the other ``eval()`` call sites (e.g. ``copilot/dream/locks``):
    the cluster client types ``eval()``'s return as ``str``.
    """
    redis = await get_redis_async()
    return await cast(
        Any,
        redis.eval(
            _INCR_OPEN_WINDOW,
            1,
            key,
            str(SUBSCRIPTION_STATUS_WINDOW_SECONDS),
        ),
    )


def _window_key(user_id: str, *, now: datetime) -> str:
    """Per-user fixed-window key, bucket-aligned to the window so each user
    has at most one active counter per window."""
    bucket = int(now.timestamp()) // SUBSCRIPTION_STATUS_WINDOW_SECONDS
    return f"credits:subscription:rl:{user_id}:{bucket}"


async def enforce_subscription_status_rate_limit(
    user_id: str = fastapi.Security(get_user_id),
) -> None:
    """Raise HTTP 429 when ``user_id`` exceeds the per-window cap.

    Wired as a route dependency on ``GET /credits/subscription`` only, so the
    internal callers of ``get_subscription_status`` (e.g. the POST update flow
    returning fresh state) never trip it — FastAPI dependencies run for HTTP
    requests, not for direct function calls.

    On any Redis trouble this fails *open* (logs and lets the call through): a
    Redis blip must never block the billing status read for every user, which
    would be far worse than one client briefly exceeding its cap. The whole
    interaction is bounded by ``SUBSCRIPTION_STATUS_REDIS_TIMEOUT_SECONDS`` so
    an outage falls through fast instead of holding a worker slot.

    ``RedisClusterException`` is caught explicitly: it does not inherit from
    ``RedisError``, and ``SlotNotCoveredError`` (raised during a cluster
    rolling restart / slot migration) would otherwise surface as a 500 — the
    exact brown-out this fail-open exists for. Same reasoning as the comment in
    ``backend/copilot/rate_limit.py``.
    """
    now = datetime.now(UTC)
    key = _window_key(user_id, now=now)
    try:
        count = await asyncio.wait_for(
            _incr_window(key),
            timeout=SUBSCRIPTION_STATUS_REDIS_TIMEOUT_SECONDS,
        )
    except (
        RedisError,
        RedisClusterException,
        ConnectionError,
        OSError,
        asyncio.TimeoutError,
        ValueError,
    ) as e:
        logger.warning(
            "Subscription-status rate-limit check failed open for user %s: %s",
            user_id,
            e,
        )
        return

    if count > SUBSCRIPTION_STATUS_MAX_REQUESTS:
        # Seconds left in this fixed window. Without Retry-After the client
        # retries blind (React Query defaults to 3 retries), turning one block
        # into several extra requests against the endpoint being protected.
        retry_after = SUBSCRIPTION_STATUS_WINDOW_SECONDS - (
            int(now.timestamp()) % SUBSCRIPTION_STATUS_WINDOW_SECONDS
        )
        logger.info(
            "Subscription-status rate limit hit for user %s (count=%s)",
            user_id,
            count,
        )
        record_rate_limit_hit("/api/credits/subscription", user_id)
        raise fastapi.HTTPException(
            status_code=429,
            detail=(
                f"Subscription status rate limit exceeded "
                f"({SUBSCRIPTION_STATUS_MAX_REQUESTS} requests per "
                f"{SUBSCRIPTION_STATUS_WINDOW_SECONDS}s). Try again shortly."
            ),
            headers={"Retry-After": str(retry_after)},
        )
