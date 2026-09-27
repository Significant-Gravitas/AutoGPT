# -*- coding: utf-8 -*-
"""Per-principal rate limiting for the AutoGPT external API.

The external API authenticates with API keys / OAuth tokens, so one valid
credential can hammer store endpoints or spam graph executions. This module
provides an atomic fixed-window counter in Redis (Lua ``INCR`` + first-hit
``EXPIRE``, mirroring ``backend/api/features/credits_rate_limit.py``) keyed
per (user, scope), with per-scope tiers, returning HTTP 429 with
``Retry-After`` / ``X-RateLimit-*`` headers once the window counter is
exhausted.

The bucket is deliberately keyed on the *user*, not on the credential: an
abuse control must not be resettable by minting another API key or by
refreshing an OAuth access token, both of which produce a new credential id
for the same principal.

Availability: the check is strictly best-effort and **fails open** on any
Redis trouble — being unable to prove a principal is under its cap must never
take the whole external API down. The whole Redis interaction is bounded by a
short deadline so an outage falls through fast instead of parking ASGI worker
slots (same reasoning as ``credits_rate_limit``).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import Depends, HTTPException, Response, Security, status
from redis.exceptions import RedisClusterException, RedisError

from backend.api.external.middleware import require_auth
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.redis_client import get_redis_async
from backend.monitoring.instrumentation import record_rate_limit_hit

logger = logging.getLogger(__name__)

# Per-scope tiers: (max requests, window seconds).
# Cheap reads (identity, block listing) get a generous cap; anything that
# executes work (block execution, graph runs) gets a tight one.
READ_MAX_REQUESTS = 300
READ_WINDOW_SECONDS = 60
EXECUTION_MAX_REQUESTS = 30
EXECUTION_WINDOW_SECONDS = 60

# Hard deadline on the whole Redis interaction. Without it "fail open" is not
# "fail fast": a cold client runs redis-py's own connect retry ladder, so
# during an outage a request would park an ASGI worker slot for minutes
# instead of falling through.
RATE_LIMIT_REDIS_TIMEOUT_SECONDS = 0.25

# Atomic fixed-window counter: INCR the key, and set the TTL only on the INCR
# that opened the window (count == 1). Doing both in one server-side script
# means a freshly-created key always gets its expiry — the key can never
# linger without a TTL, and the window stays fixed (the TTL is not refreshed
# on later hits).
_INCR_OPEN_WINDOW = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


def _rate_limit_key(auth: APIAuthorizationInfo, scope: str) -> str:
    """One bucket per (user, scope).

    The user is the unit that cannot be multiplied, so the budget must follow
    the user rather than the credential presented with the request. Minting a
    second API key, or letting an OAuth access token refresh, both yield a new
    credential ``id`` for the same principal; keying on that id would hand out
    a fresh budget each time and make the limit trivially bypassable.
    """
    return f"external_api:rl:{auth.user_id}:{scope}"


def _window_start(now: datetime, window_seconds: int) -> int:
    """Start of the wall-clock window ``now`` falls in.

    The window is aligned to absolute time so the key, ``X-RateLimit-Reset``
    and ``Retry-After`` all describe the same instant. A client that waits for
    ``Retry-After`` therefore lands in the next window and is admitted, instead
    of being refused again because the TTL runs from its first hit.
    """
    return int(now.timestamp()) // window_seconds * window_seconds


async def _incr_window(key: str, window_seconds: int) -> int:
    """Run the atomic counter, returning the request's position in the window.

    Lua ARGV values are strings; ``EXPIRE`` coerces ``"60"`` back to an int.
    The cast mirrors the other ``eval()`` call sites (e.g.
    ``credits_rate_limit``): the cluster client types ``eval()``'s return as
    ``str``.
    """
    redis = await get_redis_async()
    return await cast(
        Any,
        redis.eval(_INCR_OPEN_WINDOW, 1, key, str(window_seconds)),
    )


def require_rate_limit(
    *,
    max_requests: int,
    window_seconds: int,
    scope: str,
):
    """FastAPI dependency enforcing a per-principal rate limit on a route.

    The dependency injects the ``Response`` so it can set ``X-RateLimit-*``
    headers on the way out, and raises HTTP 429 (with ``Retry-After``) once
    the fixed-window counter is exhausted. On any Redis trouble it fails
    open: logs and lets the request through.
    """

    async def dependency(
        response: Response,
        auth: APIAuthorizationInfo = Security(require_auth),
    ) -> APIAuthorizationInfo:
        now = datetime.now(UTC)
        window_start = _window_start(now, window_seconds)
        # The window start is part of the key so that the counter, the TTL and
        # the reported reset time all refer to the same window: a request is
        # never counted against a window it is told it can retry out of.
        key = f"{_rate_limit_key(auth, scope)}:{window_start}"
        try:
            count = await asyncio.wait_for(
                _incr_window(key, window_seconds),
                timeout=RATE_LIMIT_REDIS_TIMEOUT_SECONDS,
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
                "External API rate-limit check failed open for user %s scope %s: %s",
                auth.user_id,
                scope,
                e,
            )
            return auth

        remaining = max(0, max_requests - int(count))
        reset_at = window_start + window_seconds
        retry_after = max(0, reset_at - int(now.timestamp()))

        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)

        if int(count) > max_requests:
            logger.info(
                "External API rate limit hit for user %s scope %s (count=%s)",
                auth.user_id,
                scope,
                count,
            )
            record_rate_limit_hit(f"external_api:{scope}", auth.user_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for scope '{scope}'. "
                f"Retry after {retry_after}s.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                },
            )
        return auth

    return dependency


# Ready-made dependencies for the two tiers. Every ``/v1`` route carries one
# of these, so they are defined once here rather than repeating the eight-line
# ``Depends(require_rate_limit(...))`` block at each of the twelve call sites.
READ_LIMIT = Depends(
    require_rate_limit(
        max_requests=READ_MAX_REQUESTS,
        window_seconds=READ_WINDOW_SECONDS,
        scope="read",
    )
)
EXECUTION_LIMIT = Depends(
    require_rate_limit(
        max_requests=EXECUTION_MAX_REQUESTS,
        window_seconds=EXECUTION_WINDOW_SECONDS,
        scope="execution",
    )
)
