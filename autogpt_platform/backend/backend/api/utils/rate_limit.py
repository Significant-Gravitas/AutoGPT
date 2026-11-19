"""
Redis-backed fixed-window rate limiter.

Each (limiter, user) pair gets a Redis key with TTL equal to the window;
``SET NX EX`` creates the key atomically on the first hit, then ``INCR``
counts requests within the window.

Fails **open** on Redis errors: a transient Redis blip logs a warning and
lets the request through rather than blocking all API traffic.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Every v2 request waits on these checks, so a slow Redis fails them open fast.
REDIS_TIMEOUT_SECONDS = 0.25


class RateLimitState(BaseModel):
    """Where a request left the caller's window, for the response headers."""

    limit: int
    remaining: int
    reset_seconds: int

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_seconds),
        }


class RateLimiter:
    """Redis fixed-window rate limiter."""

    def __init__(self, name: str, *, max_requests: int, window_seconds: int):
        self.name = name
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def check(self, user_id: str) -> Optional[RateLimitState]:
        """Raise HTTP 429 if the user exceeds the per-window cap.

        Returns where the request left the window, or `None` when the count is
        unknown because Redis was unreachable — the caller then publishes no
        `X-RateLimit-*` headers rather than a number it did not measure.
        """
        now = datetime.now(UTC)
        key = self._key(user_id, now)
        try:
            count = await asyncio.wait_for(
                self._count(key), timeout=REDIS_TIMEOUT_SECONDS
            )
        except Exception as e:
            # A Redis outage fails every caller open, so the subject adds nothing;
            # for the credential buckets it would also put a credential in the log.
            logger.warning(f"Rate-limit check ({self.name}) failed open: {e}")
            return None

        state = RateLimitState(
            limit=self.max_requests,
            remaining=max(0, self.max_requests - count),
            reset_seconds=self._reset_seconds(now),
        )
        if count > self.max_requests:
            # Without Retry-After a blocked client retries blind, turning one
            # block into several more requests against what the cap protects.
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded ({self.max_requests} requests "
                    f"per {self.window_seconds}s). Try again shortly."
                ),
                headers={
                    "Retry-After": str(state.reset_seconds),
                    **state.headers(),
                },
            )
        return state

    async def exhausted(self, user_id: str) -> bool:
        """Whether the window is used up, without counting this call.

        For a limiter that counts only some outcomes, checked before work the
        count exists to bound. Fails open, like ``check``.
        """
        try:
            count = await asyncio.wait_for(
                self._read(self._key(user_id, datetime.now(UTC))),
                timeout=REDIS_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(f"Rate-limit peek ({self.name}) failed open: {e}")
            return False
        return count is not None and int(count) >= self.max_requests

    async def _count(self, key: str) -> int:
        redis = await get_redis_async()
        await redis.set(key, 0, ex=self.window_seconds, nx=True)
        return await redis.incr(key)

    async def _read(self, key: str) -> Optional[bytes | str]:
        redis = await get_redis_async()
        return await redis.get(key)

    def _key(self, user_id: str, now: datetime) -> str:
        bucket = int(now.timestamp()) // self.window_seconds
        return f"rl:{self.name}:{user_id}:{bucket}"

    def _reset_seconds(self, now: datetime) -> int:
        """Seconds until this fixed window rolls over."""
        return self.window_seconds - (int(now.timestamp()) % self.window_seconds)
