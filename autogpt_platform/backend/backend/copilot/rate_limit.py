"""CoPilot rate limiting based on token usage.

Uses Redis fixed-window counters to track per-user token consumption
with configurable daily and weekly limits. Daily windows reset at
midnight UTC; weekly windows reset at ISO week boundary (Monday 00:00
UTC). Fails open when Redis is unavailable to avoid blocking users.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Redis key prefixes
_USAGE_KEY_PREFIX = "copilot:usage"


class UsageWindow(BaseModel):
    """Usage within a single time window."""

    used: int
    limit: int = Field(
        description="Maximum tokens allowed in this window. 0 means unlimited."
    )
    resets_at: datetime


class CoPilotUsageStatus(BaseModel):
    """Current usage status for a user across all windows."""

    daily: UsageWindow
    weekly: UsageWindow


class RateLimitExceeded(Exception):
    """Raised when a user exceeds their CoPilot usage limit."""

    def __init__(self, window: str, resets_at: datetime):
        self.window = window
        self.resets_at = resets_at
        delta = resets_at - datetime.now(UTC)
        total_secs = delta.total_seconds()
        if total_secs <= 0:
            time_str = "now"
        else:
            hours = int(total_secs // 3600)
            minutes = int((total_secs % 3600) // 60)
            time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        super().__init__(
            f"You've reached your {window} usage limit. Resets in {time_str}."
        )


async def get_usage_status(
    user_id: str,
    daily_token_limit: int,
    weekly_token_limit: int,
) -> CoPilotUsageStatus:
    """Get current usage status for a user.

    Args:
        user_id: The user's ID.
        daily_token_limit: Max tokens per day (0 = unlimited).
        weekly_token_limit: Max tokens per week (0 = unlimited).

    Returns:
        CoPilotUsageStatus with current usage and limits.
    """
    now = datetime.now(UTC)
    daily_used = 0
    weekly_used = 0
    try:
        redis = await get_redis_async()
        daily_raw, weekly_raw = await asyncio.gather(
            redis.get(_daily_key(user_id, now=now)),
            redis.get(_weekly_key(user_id, now=now)),
        )
        daily_used = int(daily_raw or 0)
        weekly_used = int(weekly_raw or 0)
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for usage status, returning zeros")

    return CoPilotUsageStatus(
        daily=UsageWindow(
            used=daily_used,
            limit=daily_token_limit,
            resets_at=_daily_reset_time(now=now),
        ),
        weekly=UsageWindow(
            used=weekly_used,
            limit=weekly_token_limit,
            resets_at=_weekly_reset_time(now=now),
        ),
    )


async def check_rate_limit(
    user_id: str,
    daily_token_limit: int,
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
    # Short-circuit: when both limits are 0 (unlimited) skip the Redis
    # round-trip entirely.
    if daily_token_limit <= 0 and weekly_token_limit <= 0:
        return

    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        daily_raw, weekly_raw = await asyncio.gather(
            redis.get(_daily_key(user_id, now=now)),
            redis.get(_weekly_key(user_id, now=now)),
        )
        daily_used = int(daily_raw or 0)
        weekly_used = int(weekly_raw or 0)
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for rate limit check, allowing request")
        return

    # Worst-case overshoot: N concurrent requests × ~15K tokens each.
    if daily_token_limit > 0 and daily_used >= daily_token_limit:
        raise RateLimitExceeded("daily", _daily_reset_time(now=now))

    if weekly_token_limit > 0 and weekly_used >= weekly_token_limit:
        raise RateLimitExceeded("weekly", _weekly_reset_time(now=now))


async def record_token_usage(
    user_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    """Record token usage for a user across all windows.

    Uses cost-weighted counting so cached tokens don't unfairly penalise
    multi-turn conversations. Anthropic's pricing:
      - uncached input: 100%
      - cache creation:  25%
      - cache read:      10%
      - output:         100%

    ``prompt_tokens`` should be the *uncached* input count (``input_tokens``
    from the API response). Cache counts are passed separately.

    Args:
        user_id: The user's ID.
        prompt_tokens: Uncached input tokens.
        completion_tokens: Output tokens.
        cache_read_tokens: Tokens served from prompt cache (10% cost).
        cache_creation_tokens: Tokens written to prompt cache (25% cost).
    """
    prompt_tokens = max(0, prompt_tokens)
    completion_tokens = max(0, completion_tokens)
    cache_read_tokens = max(0, cache_read_tokens)
    cache_creation_tokens = max(0, cache_creation_tokens)

    weighted_input = (
        prompt_tokens
        + round(cache_creation_tokens * 0.25)
        + round(cache_read_tokens * 0.1)
    )
    total = weighted_input + completion_tokens
    if total <= 0:
        return

    raw_total = (
        prompt_tokens + cache_read_tokens + cache_creation_tokens + completion_tokens
    )
    logger.info(
        "Recording token usage for %s: raw=%d, weighted=%d "
        "(uncached=%d, cache_read=%d@10%%, cache_create=%d@25%%, output=%d)",
        user_id[:8],
        raw_total,
        total,
        prompt_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        completion_tokens,
    )

    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        # transaction=False: these are independent INCRBY+EXPIRE pairs on
        # separate keys — no cross-key atomicity needed.  Skipping
        # MULTI/EXEC avoids the overhead.  If the connection drops between
        # INCRBY and EXPIRE the key survives until the next date-based key
        # rotation (daily/weekly), so the memory-leak risk is negligible.
        pipe = redis.pipeline(transaction=False)

        # Daily counter (expires at next midnight UTC)
        d_key = _daily_key(user_id, now=now)
        pipe.incrby(d_key, total)
        seconds_until_daily_reset = int(
            (_daily_reset_time(now=now) - now).total_seconds()
        )
        pipe.expire(d_key, max(seconds_until_daily_reset, 1))

        # Weekly counter (expires end of week)
        w_key = _weekly_key(user_id, now=now)
        pipe.incrby(w_key, total)
        seconds_until_weekly_reset = int(
            (_weekly_reset_time(now=now) - now).total_seconds()
        )
        pipe.expire(w_key, max(seconds_until_weekly_reset, 1))

        await pipe.execute()
    except (RedisError, ConnectionError, OSError):
        logger.warning(
            "Redis unavailable for recording token usage (tokens=%d)",
            total,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _daily_key(user_id: str, now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(UTC)
    return f"{_USAGE_KEY_PREFIX}:daily:{user_id}:{now.strftime('%Y-%m-%d')}"


def _weekly_key(user_id: str, now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(UTC)
    year, week, _ = now.isocalendar()
    return f"{_USAGE_KEY_PREFIX}:weekly:{user_id}:{year}-W{week:02d}"


def _daily_reset_time(now: datetime | None = None) -> datetime:
    """Calculate when the current daily window resets (next midnight UTC)."""
    if now is None:
        now = datetime.now(UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _weekly_reset_time(now: datetime | None = None) -> datetime:
    """Calculate when the current weekly window resets (next Monday 00:00 UTC)."""
    if now is None:
        now = datetime.now(UTC)
    days_until_monday = (7 - now.weekday()) % 7 or 7
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=days_until_monday
    )
