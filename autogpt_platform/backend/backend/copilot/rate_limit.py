"""CoPilot rate limiting based on token usage.

Uses Redis sliding-window counters to track per-user token consumption
with configurable session and weekly limits. Fails open when Redis is
unavailable to avoid blocking users.
"""

import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Redis key prefixes
_PREFIX = "copilot:usage"


class UsageWindow(BaseModel):
    """Usage within a single time window."""

    used: int
    limit: int
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
        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        else:
            time_str = f"{minutes}m"
        super().__init__(
            f"You've reached your {window} usage limit. " f"Resets in {time_str}."
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
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=days_until_monday
    )


def _session_reset_time() -> datetime:
    """Session limits reset after 3 hours of inactivity (matching session TTL)."""
    return datetime.now(UTC) + timedelta(hours=3)


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
        session_token_limit: Max tokens per session.
        weekly_token_limit: Max tokens per week.

    Returns:
        CoPilotUsageStatus with current usage and limits.
    """
    try:
        redis = await get_redis_async()
        session_used = int(await redis.get(_session_key(user_id, session_id)) or 0)
        weekly_used = int(await redis.get(_weekly_key(user_id)) or 0)
    except Exception:
        logger.warning("Redis unavailable for usage status, returning zeros")
        session_used = 0
        weekly_used = 0

    return CoPilotUsageStatus(
        session=UsageWindow(
            used=session_used,
            limit=session_token_limit,
            resets_at=_session_reset_time(),
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

    Fails open: if Redis is unavailable, allows the request.
    """
    try:
        redis = await get_redis_async()
        session_used = int(await redis.get(_session_key(user_id, session_id)) or 0)
        weekly_used = int(await redis.get(_weekly_key(user_id)) or 0)
    except Exception:
        logger.warning("Redis unavailable for rate limit check, allowing request")
        return

    if session_token_limit > 0 and session_used >= session_token_limit:
        raise RateLimitExceeded("session", _session_reset_time())

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
        pipe = redis.pipeline()

        # Session counter (reset with session TTL — 12 hours)
        s_key = _session_key(user_id, session_id)
        pipe.incrby(s_key, total)
        pipe.expire(s_key, 43200)  # 12 hours

        # Weekly counter (expires end of week)
        w_key = _weekly_key(user_id)
        pipe.incrby(w_key, total)
        seconds_until_reset = int(
            (_weekly_reset_time() - datetime.now(UTC)).total_seconds()
        )
        pipe.expire(w_key, max(seconds_until_reset, 1))

        await pipe.execute()
    except Exception:
        logger.warning(
            "Redis unavailable for recording token usage (user=%s, tokens=%d)",
            user_id,
            total,
        )
