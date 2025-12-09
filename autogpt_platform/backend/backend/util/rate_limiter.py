"""
Rate Limiting for External API.

Implements sliding window rate limiting using Redis for distributed systems.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


class RateLimiter:
    """
    Redis-based sliding window rate limiter.

    Supports multiple limit tiers (per-minute, per-hour, per-day).
    """

    def __init__(self, prefix: str = "ratelimit"):
        self.prefix = prefix

    def _make_key(self, identifier: str, window: str) -> str:
        """Create a Redis key for the rate limit counter."""
        return f"{self.prefix}:{identifier}:{window}"

    async def check_and_increment(
        self,
        identifier: str,
        limits: dict[str, tuple[int, int]],  # window_name -> (limit, window_seconds)
    ) -> RateLimitResult:
        """
        Check rate limits and increment counters if allowed.

        Uses atomic increment-first approach to prevent race conditions:
        1. Increment all counters atomically
        2. Check if any limit exceeded
        3. If exceeded, decrement and return rate limit error

        Args:
            identifier: Unique identifier (e.g., client_id, client_id:user_id)
            limits: Dictionary of limit configurations
                    e.g., {"minute": (60, 60), "hour": (1000, 3600)}

        Returns:
            RateLimitResult with allowed status and remaining quota
        """
        if not limits:
            # No limits configured, allow request
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_at=time.time() + 60,
            )

        redis = await get_redis_async()
        current_time = time.time()

        # Increment all counters atomically first
        incremented_keys: list[tuple[str, int, int, int]] = (
            []
        )  # (key, new_count, limit, window_seconds)

        for window_name, (limit, window_seconds) in limits.items():
            key = self._make_key(identifier, window_name)

            # Atomic increment
            new_count = await redis.incr(key)

            # Set expiry if this is a new key
            if new_count == 1:
                await redis.expire(key, window_seconds)

            incremented_keys.append((key, new_count, limit, window_seconds))

        # Check if any limit exceeded
        for key, new_count, limit, window_seconds in incremented_keys:
            if new_count > limit:
                # Rate limit exceeded - decrement all counters we just incremented
                for decr_key, _, _, _ in incremented_keys:
                    await redis.decr(decr_key)

                ttl = await redis.ttl(key)
                reset_at = current_time + (ttl if ttl > 0 else window_seconds)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=ttl if ttl > 0 else window_seconds,
                )

        # All limits passed
        min_remaining = float("inf")
        earliest_reset = current_time

        for key, new_count, limit, window_seconds in incremented_keys:
            remaining = max(0, limit - new_count)
            min_remaining = min(min_remaining, remaining)

            ttl = await redis.ttl(key)
            reset_at = current_time + (ttl if ttl > 0 else window_seconds)
            earliest_reset = max(earliest_reset, reset_at)

        return RateLimitResult(
            allowed=True,
            remaining=int(min_remaining),
            reset_at=earliest_reset,
        )

    async def get_remaining(
        self,
        identifier: str,
        limits: dict[str, tuple[int, int]],
    ) -> dict[str, int]:
        """
        Get remaining quota for all windows without incrementing.

        Args:
            identifier: Unique identifier
            limits: Dictionary of limit configurations

        Returns:
            Dictionary of remaining quota per window
        """
        redis = await get_redis_async()
        remaining = {}

        for window_name, (limit, _) in limits.items():
            key = self._make_key(identifier, window_name)
            count = await redis.get(key)
            current_count = int(count) if count else 0
            remaining[window_name] = max(0, limit - current_count)

        return remaining

    async def reset(self, identifier: str, window: Optional[str] = None) -> None:
        """
        Reset rate limit counters.

        Args:
            identifier: Unique identifier
            window: Optional specific window to reset (resets all if None)
        """
        redis = await get_redis_async()

        if window:
            key = self._make_key(identifier, window)
            await redis.delete(key)
        else:
            # Delete known window keys instead of scanning
            # This avoids potentially slow scan operations with many keys
            known_windows = ["minute", "hour", "day"]
            keys_to_delete = [self._make_key(identifier, w) for w in known_windows]
            # Delete all in one call (Redis handles non-existent keys gracefully)
            if keys_to_delete:
                await redis.delete(*keys_to_delete)


# Default rate limits for different endpoints
DEFAULT_RATE_LIMITS = {
    # OAuth endpoints
    "oauth_authorize": {"minute": (30, 60)},  # 30/min per IP
    "oauth_token": {"minute": (20, 60)},  # 20/min per client
    # External API endpoints
    "api_execute": {
        "minute": (10, 60),
        "hour": (100, 3600),
    },  # 10/min, 100/hour per client+user
    "api_read": {
        "minute": (60, 60),
        "hour": (1000, 3600),
    },  # 60/min, 1000/hour per client+user
}


# Module-level singleton
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the singleton rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def check_rate_limit(
    identifier: str,
    limit_type: str,
) -> RateLimitResult:
    """
    Convenience function to check rate limits.

    Args:
        identifier: Unique identifier for the rate limit
        limit_type: Type of limit from DEFAULT_RATE_LIMITS

    Returns:
        RateLimitResult
    """
    limits = DEFAULT_RATE_LIMITS.get(limit_type)
    if not limits:
        # No rate limit configured, allow
        return RateLimitResult(
            allowed=True,
            remaining=999999,
            reset_at=time.time() + 60,
        )

    rate_limiter = get_rate_limiter()
    return await rate_limiter.check_and_increment(identifier, limits)
