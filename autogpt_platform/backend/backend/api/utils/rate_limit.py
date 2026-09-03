"""
Redis-backed fixed-window rate limiter.

Each (limiter, user) pair gets a Redis key with TTL equal to the window;
``SET NX EX`` creates the key atomically on the first hit, then ``INCR``
counts requests within the window.

Fails **open** on Redis errors: a transient Redis blip logs a warning and
lets the request through rather than blocking all API traffic.
"""

import logging
from datetime import UTC, datetime

from fastapi import HTTPException

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis fixed-window rate limiter."""

    def __init__(self, name: str, *, max_requests: int, window_seconds: int):
        self.name = name
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def _key(self, user_id: str, now: datetime) -> str:
        bucket = int(now.timestamp()) // self.window_seconds
        return f"rl:{self.name}:{user_id}:{bucket}"

    async def check(self, user_id: str) -> None:
        """Raise HTTP 429 if the user exceeds the per-window cap."""
        now = datetime.now(UTC)
        key = self._key(user_id, now)
        try:
            redis = await get_redis_async()
            await redis.set(key, 0, ex=self.window_seconds, nx=True)
            count = await redis.incr(key)
        except Exception as e:
            logger.warning(
                "Rate-limit check (%s) failed open for user %s: %s",
                self.name,
                user_id,
                e,
            )
            return

        if count > self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded ({self.max_requests} requests "
                    f"per {self.window_seconds}s). Try again shortly."
                ),
            )
