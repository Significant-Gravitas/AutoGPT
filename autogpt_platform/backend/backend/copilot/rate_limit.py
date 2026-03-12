"""CoPilot rate limiting based on token usage.

Uses Redis fixed-window counters to track per-user token consumption
with configurable session and weekly limits. Session windows reset after
a 12-hour inactivity TTL; weekly windows reset at ISO week boundary
(Monday 00:00 UTC). Fails open when Redis is unavailable to avoid
blocking users.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Redis key prefixes
_PREFIX = "copilot:usage"

# Session keys expire after 12 hours of inactivity
_SESSION_TTL_SECONDS = 43200  # 12 hours


class UsageWindow(BaseModel):
    """Usage within a single time window."""

    used: int
    limit: int = Field(
        description="Maximum tokens allowed in this window. 0 means unlimited."
    )
    resets_at: datetime


class CoPilotUsageStatus(BaseModel):
    """Current usage status for a user across all windows."""

    session: UsageWindow
    weekly: UsageWindow


class RateLimitExceeded(Exception):
    """Raised when a user exceeds their CoPilot usage limit."""

    def __init__(self, window: str, resets_at: datetime):
        self.window = window
        self.resets_at = resets_at
        delta = resets_at - datetime.now(UTC)
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        super().__init__(
            f"You've reached your {window} usage limit. Resets in {time_str}."
        )


def _session_key(user_id: str, session_id: str) -> str:
    return f"{_PREFIX}:session:{user_id}:{session_id}"


def _weekly_key(user_id: str) -> str:
    now = datetime.now(UTC)
    # ISO week number
    year, week, _ = now.isocalendar()
    return f"{_PREFIX}:weekly:{user_id}:{year}-W{week:02d}"


def _weekly_reset_time() -> datetime:
    """Calculate when the current weekly window resets (next Monday 00:00 UTC)."""
    now = datetime.now(UTC)
    days_until_monday = (7 - now.weekday()) % 7 or 7
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=days_until_monday
    )


async def _session_reset_from_ttl(
    redis: AsyncRedis, user_id: str, session_id: str  # type: ignore[type-arg]
) -> datetime:
    """Derive session reset time from the Redis key's actual TTL.

    Falls back to the configured TTL if the key doesn't exist or has no expiry.
    """
    try:
        ttl: int = await redis.ttl(_session_key(user_id, session_id))
        if ttl > 0:
            return datetime.now(UTC) + timedelta(seconds=ttl)
    except (RedisError, ConnectionError, OSError):
        pass
    # Key doesn't exist or has no TTL — use the configured TTL
    return datetime.now(UTC) + timedelta(seconds=_SESSION_TTL_SECONDS)


async def get_usage_status(
    user_id: str,
    session_id: str,
    session_token_limit: int,
    weekly_token_limit: int,
) -> CoPilotUsageStatus:
    """Get current usage status for a user.

    Args:
        user_id: The user's ID.
        session_id: The current session ID.
        session_token_limit: Max tokens per session (0 = unlimited).
        weekly_token_limit: Max tokens per week (0 = unlimited).

    Returns:
        CoPilotUsageStatus with current usage and limits.
    """
    session_resets_at = datetime.now(UTC) + timedelta(seconds=_SESSION_TTL_SECONDS)
    try:
        redis = await get_redis_async()
        session_raw, weekly_raw = await asyncio.gather(
            redis.get(_session_key(user_id, session_id)),
            redis.get(_weekly_key(user_id)),
        )
        session_used = int(session_raw or 0)
        weekly_used = int(weekly_raw or 0)
        session_resets_at = await _session_reset_from_ttl(redis, user_id, session_id)
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for usage status, returning zeros")
        session_used = 0
        weekly_used = 0

    return CoPilotUsageStatus(
        session=UsageWindow(
            used=session_used,
            limit=session_token_limit,
            resets_at=session_resets_at,
        ),
        weekly=UsageWindow(
            used=weekly_used,
            limit=weekly_token_limit,
            resets_at=_weekly_reset_time(),
        ),
    )


async def check_rate_limit(
    user_id: str,
    session_id: str,
    session_token_limit: int,
    weekly_token_limit: int,
) -> None:
    """Check if user is within rate limits. Raises RateLimitExceeded if not.

    This is a pre-turn soft check. The authoritative usage counter is updated
    by ``record_token_usage()`` after the turn completes. Under concurrency,
    two parallel turns may both pass this check against the same snapshot.
    This is acceptable because token-based limits are approximate by nature
    (the exact token count is unknown until after generation).

    Fails open: if Redis is unavailable, allows the request.
    """
    try:
        redis = await get_redis_async()
        session_raw, weekly_raw = await asyncio.gather(
            redis.get(_session_key(user_id, session_id)),
            redis.get(_weekly_key(user_id)),
        )
        session_used = int(session_raw or 0)
        weekly_used = int(weekly_raw or 0)
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for rate limit check, allowing request")
        return

    if session_token_limit > 0 and session_used >= session_token_limit:
        resets_at = await _session_reset_from_ttl(redis, user_id, session_id)
        raise RateLimitExceeded("session", resets_at)

    if weekly_token_limit > 0 and weekly_used >= weekly_token_limit:
        raise RateLimitExceeded("weekly", _weekly_reset_time())


async def record_token_usage(
    user_id: str,
    session_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Record token usage for a user across all windows.

    Args:
        user_id: The user's ID.
        session_id: The current session ID.
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens used.
    """
    total = prompt_tokens + completion_tokens
    if total <= 0:
        return

    try:
        redis = await get_redis_async()
        pipe = redis.pipeline(transaction=False)

        # Session counter (expires after configured TTL)
        s_key = _session_key(user_id, session_id)
        pipe.incrby(s_key, total)
        pipe.expire(s_key, _SESSION_TTL_SECONDS)

        # Weekly counter (expires end of week)
        w_key = _weekly_key(user_id)
        pipe.incrby(w_key, total)
        seconds_until_reset = int(
            (_weekly_reset_time() - datetime.now(UTC)).total_seconds()
        )
        pipe.expire(w_key, max(seconds_until_reset, 1))

        await pipe.execute()
    except (RedisError, ConnectionError, OSError):
        logger.warning(
            "Redis unavailable for recording token usage (tokens=%d)",
            total,
        )
