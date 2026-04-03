"""CoPilot rate limiting based on token usage.

Uses Redis fixed-window counters to track per-user token consumption
with configurable daily and weekly limits. Daily windows reset at
midnight UTC; weekly windows reset at ISO week boundary (Monday 00:00
UTC). Fails open when Redis is unavailable to avoid blocking users.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from enum import Enum

from prisma.models import User as PrismaUser
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from backend.data.redis_client import get_redis_async
from backend.util.cache import cached

logger = logging.getLogger(__name__)

# Redis key prefixes
_USAGE_KEY_PREFIX = "copilot:usage"


# ---------------------------------------------------------------------------
# Subscription tier definitions
# ---------------------------------------------------------------------------


class SubscriptionTier(str, Enum):
    """Subscription tiers with increasing token allowances.

    Mirrors the ``SubscriptionTier`` enum in ``schema.prisma``.
    Once ``prisma generate`` is run, this can be replaced with::

        from prisma.enums import SubscriptionTier
    """

    FREE = "FREE"
    PRO = "PRO"
    BUSINESS = "BUSINESS"
    ENTERPRISE = "ENTERPRISE"


# Multiplier applied to the base limits (from LD / config) for each tier.
# Intentionally int (not float): keeps limits as whole token counts and avoids
# floating-point rounding.  If fractional multipliers are ever needed, change
# the type and round the result in get_global_rate_limits().
TIER_MULTIPLIERS: dict[SubscriptionTier, int] = {
    SubscriptionTier.FREE: 1,
    SubscriptionTier.PRO: 5,
    SubscriptionTier.BUSINESS: 20,
    SubscriptionTier.ENTERPRISE: 60,
}

DEFAULT_TIER = SubscriptionTier.FREE


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
    tier: SubscriptionTier = DEFAULT_TIER
    reset_cost: int = Field(
        default=0,
        description="Credit cost (in cents) to reset the daily limit. 0 = feature disabled.",
    )


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
    rate_limit_reset_cost: int = 0,
    tier: SubscriptionTier = DEFAULT_TIER,
) -> CoPilotUsageStatus:
    """Get current usage status for a user.

    Args:
        user_id: The user's ID.
        daily_token_limit: Max tokens per day (0 = unlimited).
        weekly_token_limit: Max tokens per week (0 = unlimited).
        rate_limit_reset_cost: Credit cost (cents) to reset daily limit (0 = disabled).
        tier: The user's rate-limit tier (included in the response).

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
        tier=tier,
        reset_cost=rate_limit_reset_cost,
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


async def reset_daily_usage(user_id: str, daily_token_limit: int = 0) -> bool:
    """Reset a user's daily token usage counter in Redis.

    Called after a user pays credits to extend their daily limit.
    Also reduces the weekly usage counter by ``daily_token_limit`` tokens
    (clamped to 0) so the user effectively gets one extra day's worth of
    weekly capacity.

    Args:
        user_id: The user's ID.
        daily_token_limit: The configured daily token limit. When positive,
            the weekly counter is reduced by this amount.

    Returns False if Redis is unavailable so the caller can handle
    compensation (fail-closed for billed operations, unlike the read-only
    rate-limit checks which fail-open).
    """
    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()

        # Use a MULTI/EXEC transaction so that DELETE (daily) and DECRBY
        # (weekly) either both execute or neither does.  This prevents the
        # scenario where the daily counter is cleared but the weekly
        # counter is not decremented — which would let the caller refund
        # credits even though the daily limit was already reset.
        d_key = _daily_key(user_id, now=now)
        w_key = _weekly_key(user_id, now=now) if daily_token_limit > 0 else None

        pipe = redis.pipeline(transaction=True)
        pipe.delete(d_key)
        if w_key is not None:
            pipe.decrby(w_key, daily_token_limit)
        results = await pipe.execute()

        # Clamp negative weekly counter to 0 (best-effort; not critical).
        if w_key is not None:
            new_val = results[1]  # DECRBY result
            if new_val < 0:
                await redis.set(w_key, 0, keepttl=True)

        logger.info("Reset daily usage for user %s", user_id[:8])
        return True
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for resetting daily usage")
        return False


_RESET_LOCK_PREFIX = "copilot:reset_lock"
_RESET_COUNT_PREFIX = "copilot:reset_count"


async def acquire_reset_lock(user_id: str, ttl_seconds: int = 10) -> bool:
    """Acquire a short-lived lock to serialize rate limit resets per user."""
    try:
        redis = await get_redis_async()
        key = f"{_RESET_LOCK_PREFIX}:{user_id}"
        return bool(await redis.set(key, "1", nx=True, ex=ttl_seconds))
    except (RedisError, ConnectionError, OSError) as exc:
        logger.warning("Redis unavailable for reset lock, rejecting reset: %s", exc)
        return False


async def release_reset_lock(user_id: str) -> None:
    """Release the per-user reset lock."""
    try:
        redis = await get_redis_async()
        await redis.delete(f"{_RESET_LOCK_PREFIX}:{user_id}")
    except (RedisError, ConnectionError, OSError):
        pass  # Lock will expire via TTL


async def get_daily_reset_count(user_id: str) -> int | None:
    """Get how many times the user has reset today.

    Returns None when Redis is unavailable so callers can fail-closed
    for billed operations (as opposed to failing open for read-only
    rate-limit checks).
    """
    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        key = f"{_RESET_COUNT_PREFIX}:{user_id}:{now.strftime('%Y-%m-%d')}"
        val = await redis.get(key)
        return int(val or 0)
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for reading daily reset count")
        return None


async def increment_daily_reset_count(user_id: str) -> None:
    """Increment and track how many resets this user has done today."""
    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        key = f"{_RESET_COUNT_PREFIX}:{user_id}:{now.strftime('%Y-%m-%d')}"
        pipe = redis.pipeline(transaction=True)
        pipe.incr(key)
        seconds_until_reset = int((_daily_reset_time(now=now) - now).total_seconds())
        pipe.expire(key, max(seconds_until_reset, 1))
        await pipe.execute()
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for tracking reset count")


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


class _UserNotFoundError(Exception):
    """Raised when a user record is missing or has no subscription tier.

    Used internally by ``_fetch_user_tier`` to signal a cache-miss condition:
    by raising instead of returning ``DEFAULT_TIER``, we prevent the ``@cached``
    decorator from storing the fallback value.  This avoids a race condition
    where a non-existent user's DEFAULT_TIER is cached, then the user is
    created with a higher tier but receives the stale cached FREE tier for
    up to 5 minutes.
    """


@cached(maxsize=1000, ttl_seconds=300, shared_cache=True)
async def _fetch_user_tier(user_id: str) -> SubscriptionTier:
    """Fetch the user's rate-limit tier from the database (cached via Redis).

    Uses ``shared_cache=True`` so that tier changes propagate across all pods
    immediately when the cache entry is invalidated (via ``cache_delete``).

    Only successful DB lookups of existing users with a valid tier are cached.
    Raises ``_UserNotFoundError`` when the user is missing or has no tier, so
    the ``@cached`` decorator does **not** store a fallback value.  This
    prevents a race condition where a non-existent user's ``DEFAULT_TIER`` is
    cached and then persists after the user is created with a higher tier.
    """
    user = await PrismaUser.prisma().find_unique(where={"id": user_id})
    if user and user.subscriptionTier:  # type: ignore[reportAttributeAccessIssue]
        return SubscriptionTier(user.subscriptionTier)  # type: ignore[reportAttributeAccessIssue]
    raise _UserNotFoundError(user_id)


async def get_user_tier(user_id: str) -> SubscriptionTier:
    """Look up the user's rate-limit tier from the database.

    Successful results are cached for 5 minutes (via ``_fetch_user_tier``)
    to avoid a DB round-trip on every rate-limit check.

    Falls back to ``DEFAULT_TIER`` **without caching** when the DB is
    unreachable or returns an unrecognised value, so the next call retries
    the query instead of serving a stale fallback for up to 5 minutes.
    """
    try:
        return await _fetch_user_tier(user_id)
    except Exception as exc:
        logger.warning(
            "Failed to resolve rate-limit tier for user %s, defaulting to %s: %s",
            user_id[:8],
            DEFAULT_TIER.value,
            exc,
        )
    return DEFAULT_TIER


# Expose cache management on the public function so callers (including tests)
# never need to reach into the private ``_fetch_user_tier``.
get_user_tier.cache_clear = _fetch_user_tier.cache_clear  # type: ignore[attr-defined]
get_user_tier.cache_delete = _fetch_user_tier.cache_delete  # type: ignore[attr-defined]


async def set_user_tier(user_id: str, tier: SubscriptionTier) -> None:
    """Persist the user's rate-limit tier to the database.

    Also invalidates the ``get_user_tier`` cache for this user so that
    subsequent rate-limit checks immediately see the new tier.

    Raises:
        prisma.errors.RecordNotFoundError: If the user does not exist.
    """
    await PrismaUser.prisma().update(
        where={"id": user_id},
        data={"subscriptionTier": tier.value},
    )
    # Invalidate cached tier so rate-limit checks pick up the change immediately.
    get_user_tier.cache_delete(user_id)  # type: ignore[attr-defined]


async def get_global_rate_limits(
    user_id: str,
    config_daily: int,
    config_weekly: int,
) -> tuple[int, int, SubscriptionTier]:
    """Resolve global rate limits from LaunchDarkly, falling back to config.

    The base limits (from LD or config) are multiplied by the user's
    tier multiplier so that higher tiers receive proportionally larger
    allowances.

    Args:
        user_id: User ID for LD flag evaluation context.
        config_daily: Fallback daily limit from ChatConfig.
        config_weekly: Fallback weekly limit from ChatConfig.

    Returns:
        (daily_token_limit, weekly_token_limit, tier) 3-tuple.
    """
    # Lazy import to avoid circular dependency:
    # rate_limit -> feature_flag -> settings -> ... -> rate_limit
    from backend.util.feature_flag import Flag, get_feature_flag_value

    daily_raw = await get_feature_flag_value(
        Flag.COPILOT_DAILY_TOKEN_LIMIT.value, user_id, config_daily
    )
    weekly_raw = await get_feature_flag_value(
        Flag.COPILOT_WEEKLY_TOKEN_LIMIT.value, user_id, config_weekly
    )
    try:
        daily = max(0, int(daily_raw))
    except (TypeError, ValueError):
        logger.warning("Invalid LD value for daily token limit: %r", daily_raw)
        daily = config_daily
    try:
        weekly = max(0, int(weekly_raw))
    except (TypeError, ValueError):
        logger.warning("Invalid LD value for weekly token limit: %r", weekly_raw)
        weekly = config_weekly

    # Apply tier multiplier
    tier = await get_user_tier(user_id)
    multiplier = TIER_MULTIPLIERS.get(tier, 1)
    if multiplier != 1:
        daily = daily * multiplier
        weekly = weekly * multiplier

    return daily, weekly, tier


async def reset_user_usage(user_id: str, *, reset_weekly: bool = False) -> None:
    """Reset a user's usage counters.

    Always deletes the daily Redis key.  When *reset_weekly* is ``True``,
    the weekly key is deleted as well.

    Unlike read paths (``get_usage_status``, ``check_rate_limit``) which
    fail-open on Redis errors, resets intentionally re-raise so the caller
    knows the operation did not succeed.  A silent failure here would leave
    the admin believing the counters were zeroed when they were not.
    """
    now = datetime.now(UTC)
    keys_to_delete = [_daily_key(user_id, now=now)]
    if reset_weekly:
        keys_to_delete.append(_weekly_key(user_id, now=now))
    try:
        redis = await get_redis_async()
        await redis.delete(*keys_to_delete)
    except (RedisError, ConnectionError, OSError):
        logger.warning("Redis unavailable for resetting user usage")
        raise


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
