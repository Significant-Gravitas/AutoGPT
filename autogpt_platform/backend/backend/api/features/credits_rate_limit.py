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

import fastapi
from autogpt_libs.auth import get_user_id

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# 60 requests / minute / user — roughly 4x the heaviest realistic legitimate
# burst (a multi-tab simultaneous hard-reload is ~15/min); the shared React
# Query entry plus 60s staleTime keep steady-state use near ~1/min per tab.
SUBSCRIPTION_STATUS_WINDOW_SECONDS = 60
SUBSCRIPTION_STATUS_MAX_REQUESTS = 60


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

    On Redis brown-out this fails *open* (logs and lets the call through): a
    Redis blip must never block the billing status read for every user, which
    would be far worse than one client briefly exceeding its cap.
    """
    now = datetime.now(UTC)
    key = _window_key(user_id, now=now)
    try:
        redis = await get_redis_async()
        # Atomic create-with-TTL, then INCR. SET NX EX makes the TTL part of
        # the same write that creates the key; later INCRs preserve it.
        await redis.set(key, 0, ex=SUBSCRIPTION_STATUS_WINDOW_SECONDS, nx=True)
        count = await redis.incr(key)
    except Exception as e:
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
