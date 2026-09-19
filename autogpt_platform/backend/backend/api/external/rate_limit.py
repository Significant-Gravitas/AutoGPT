# -*- coding: utf-8 -*-
"""Per-principal rate limiting for the AutoGPT external API.

The external API authenticates with API keys / OAuth tokens, so one valid
credential can hammer store endpoints or spam graph executions. This module
provides an atomic fixed-window counter in Redis (Lua ``INCR`` + first-hit
``EXPIRE``, mirroring ``backend/api/features/credits_rate_limit.py``) keyed
per (user, credential, scope), with per-scope tiers, returning HTTP 429 with
``Retry-After`` / ``X-RateLimit-*`` headers once the window counter is
exhausted.

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

import fastapi
from fastapi import HTTPException, Response, Security, status
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
    """One bucket per (user, credential, scope).

    API-key principals carry a key ``id``, so two keys belonging to the same
    user get independent budgets; OAuth principals have no key id and fall
    back to ``user_id + type``.
    """
    principal = getattr(auth, "id", auth.type)
    return f"external_api:rl:{auth.user_id}:{principal}:{scope}"


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
        key = _rate_limit_key(auth, scope)
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

        now = datetime.now(UTC)
        remaining = max(0, max_requests - int(count))
        reset_at = (
            int(now.timestamp()) // window_seconds * window_seconds + window_seconds
        )
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
