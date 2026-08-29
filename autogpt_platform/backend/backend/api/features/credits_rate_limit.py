"""Per-user rate limit for ``GET /api/credits/subscription``.

The subscription-status endpoint is fetched on essentially every authenticated
page load (PaywallGate wraps the app shell) and, on a cold cache, fans out to
several Stripe reads per request. A per-user cap puts a hard ceiling on scripted
clients hammering it, without affecting normal use: the frontend shares one
React Query entry with a 60s ``staleTime``, so a real user — even with several
tabs open and the occasional hard reload — stays around ~5-15 requests/minute.

Fixed-window counter in Redis (``SET NX EX`` + ``INCR``), keyed per ``user_id``,
fail-open on Redis brown-out. Mirrors ``backend/api/features/search/rate_limit``.
A single key per check keeps it correct on the Redis cluster.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

import fastapi
from autogpt_libs.auth import get_user_id

from backend.data.redis_client import TRANSIENT_REDIS_ERRORS, get_redis_async

logger = logging.getLogger(__name__)

# 60 requests / minute / user — roughly 4x the heaviest realistic legitimate
# burst (a multi-tab simultaneous hard-reload is ~15/min); the shared React
# Query entry plus 60s staleTime keep steady-state use near ~1/min per tab.
SUBSCRIPTION_STATUS_WINDOW_SECONDS = 60
SUBSCRIPTION_STATUS_MAX_REQUESTS = 60

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

    On a *transient* Redis failure (connection/timeout/cluster-down) this fails
    *open* (logs and lets the call through): a Redis blip must never block the
    billing status read for every user, which would be far worse than one
    client briefly exceeding its cap. Non-transient errors are left to
    propagate rather than silently disabling the limiter.
    """
    now = datetime.now(UTC)
    key = _window_key(user_id, now=now)
    try:
        redis = await get_redis_async()
        # Lua ARGV values are strings; EXPIRE coerces "60" back to an int. The
        # cast mirrors the other eval() call sites (e.g. copilot/dream/locks):
        # the cluster client types eval()'s return as str, so cast before await.
        count = await cast(
            Any,
            redis.eval(
                _INCR_OPEN_WINDOW,
                1,
                key,
                str(SUBSCRIPTION_STATUS_WINDOW_SECONDS),
            ),
        )
    except TRANSIENT_REDIS_ERRORS as e:
        logger.warning(
            "Subscription-status rate-limit check failed open for user %s: %s",
            user_id,
            e,
        )
        return

    if count > SUBSCRIPTION_STATUS_MAX_REQUESTS:
        raise fastapi.HTTPException(
            status_code=429,
            detail=(
                f"Subscription status rate limit exceeded "
                f"({SUBSCRIPTION_STATUS_MAX_REQUESTS} requests per "
                f"{SUBSCRIPTION_STATUS_WINDOW_SECONDS}s). Try again shortly."
            ),
        )
