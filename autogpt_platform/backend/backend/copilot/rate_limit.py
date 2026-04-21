"""CoPilot rate limiting based on generation cost.

Uses Redis fixed-window counters to track per-user USD spend (stored as
microdollars, matching ``PlatformCostLog.cost_microdollars``) with
configurable daily and weekly limits. Daily windows reset at midnight UTC;
weekly windows reset at ISO week boundary (Monday 00:00 UTC). Fails open
when Redis is unavailable to avoid blocking users.

Storing microdollars rather than tokens means the counter already reflects
real model pricing (including cache discounts and provider surcharges), so
this module carries no pricing table — the cost comes from OpenRouter's
``usage.cost`` field (baseline) or the Claude Agent SDK's reported total
cost (SDK path).
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from enum import Enum

from prisma.models import User as PrismaUser
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from backend.data.db_accessors import user_db
from backend.data.redis_client import get_redis_async
from backend.data.user import get_user_by_id
from backend.util.cache import cached

logger = logging.getLogger(__name__)

# Redis key prefixes. Bumped from "copilot:usage" (token-based) to
# "copilot:cost" on the token→cost migration so stale counters do not
# get misinterpreted as microdollars (which would dramatically under-count).
_USAGE_KEY_PREFIX = "copilot:cost"


# ---------------------------------------------------------------------------
# Subscription tier definitions
# ---------------------------------------------------------------------------


class SubscriptionTier(str, Enum):
    """Subscription tiers with increasing cost allowances.

    Mirrors the ``SubscriptionTier`` enum in ``schema.prisma``.
    Once ``prisma generate`` is run, this can be replaced with::

        from prisma.enums import SubscriptionTier
    """

    FREE = "FREE"
    PRO = "PRO"
    BUSINESS = "BUSINESS"
    ENTERPRISE = "ENTERPRISE"


# Multiplier applied to the base cost limits (from LD / config) for each tier.
# Intentionally int (not float): keeps limits as whole microdollars and avoids
# floating-point rounding. If fractional multipliers are ever needed, change
# the type and round the result in get_global_rate_limits().
TIER_MULTIPLIERS: dict[SubscriptionTier, int] = {
    SubscriptionTier.FREE: 1,
    SubscriptionTier.PRO: 5,
    SubscriptionTier.BUSINESS: 20,
    SubscriptionTier.ENTERPRISE: 60,
}

DEFAULT_TIER = SubscriptionTier.FREE


class UsageWindow(BaseModel):
    """Usage within a single time window.

    ``used`` and ``limit`` are in microdollars (1 USD = 1_000_000).
    """

    used: int
    limit: int = Field(
        description="Maximum microdollars of spend allowed in this window. "
        "0 means unlimited."
    )
    resets_at: datetime


class CoPilotUsageStatus(BaseModel):
    """Current usage status for a user across all windows.

    Internal representation used by server-side code that needs to compare
    usage against limits (e.g. the reset-credits endpoint).  The public API
    returns ``CoPilotUsagePublic`` instead so that raw spend and limit
    figures never leak to clients.
    """

    daily: UsageWindow
    weekly: UsageWindow
    tier: SubscriptionTier = DEFAULT_TIER
    reset_cost: int = Field(
        default=0,
        description="Credit cost (in cents) to reset the daily limit. 0 = feature disabled.",
    )


class UsageWindowPublic(BaseModel):
    """Public view of a usage window — only the percentage and reset time.

    Hides the raw spend and the cap so clients cannot derive per-turn cost
    or reverse-engineer platform margins.  ``percent_used`` is capped at 100.
    """

    percent_used: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of the window's allowance used (0-100). "
        "Clamped at 100 when over the cap.",
    )
    resets_at: datetime


class CoPilotUsagePublic(BaseModel):
    """Current usage status for a user — public (client-safe) shape."""

    daily: UsageWindowPublic | None = Field(
        default=None,
        description="Null when no daily cap is configured (unlimited).",
    )
    weekly: UsageWindowPublic | None = Field(
        default=None,
        description="Null when no weekly cap is configured (unlimited).",
    )
    tier: SubscriptionTier = DEFAULT_TIER
    reset_cost: int = Field(
        default=0,
        description="Credit cost (in cents) to reset the daily limit. 0 = feature disabled.",
    )

    @classmethod
    def from_status(cls, status: CoPilotUsageStatus) -> "CoPilotUsagePublic":
        """Project the internal status onto the client-safe schema."""

        def window(w: UsageWindow) -> UsageWindowPublic | None:
            if w.limit <= 0:
                return None
            # When at/over the cap, snap to exactly 100.0 so the UI's
            # rounded display and its exhaustion check (`percent_used >= 100`)
            # agree. Without this, e.g. 99.95% would render as "100% used"
            # via Math.round but fail the exhaustion check, leaving the
            # reset button hidden while the bar appears full.
            if w.used >= w.limit:
                pct = 100.0
            else:
                pct = round(100.0 * w.used / w.limit, 1)
            return UsageWindowPublic(
                percent_used=pct,
                resets_at=w.resets_at,
            )

        return cls(
            daily=window(status.daily),
            weekly=window(status.weekly),
            tier=status.tier,
            reset_cost=status.reset_cost,
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
    daily_cost_limit: int,
    weekly_cost_limit: int,
    rate_limit_reset_cost: int = 0,
    tier: SubscriptionTier = DEFAULT_TIER,
) -> CoPilotUsageStatus:
    """Get current usage status for a user.

    Args:
        user_id: The user's ID.
        daily_cost_limit: Max microdollars of spend per day (0 = unlimited).
        weekly_cost_limit: Max microdollars of spend per week (0 = unlimited).
        rate_limit_reset_cost: Credit cost (cents) to reset daily limit (0 = disabled).
        tier: The user's rate-limit tier (included in the response).

    Returns:
        CoPilotUsageStatus with current usage and limits in microdollars.
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
            limit=daily_cost_limit,
            resets_at=_daily_reset_time(now=now),
        ),
        weekly=UsageWindow(
            used=weekly_used,
            limit=weekly_cost_limit,
            resets_at=_weekly_reset_time(now=now),
        ),
        tier=tier,
        reset_cost=rate_limit_reset_cost,
    )


async def check_rate_limit(
    user_id: str,
    daily_cost_limit: int,
    weekly_cost_limit: int,
) -> None:
    """Check if user is within rate limits. Raises RateLimitExceeded if not.

    This is a pre-turn soft check. The authoritative usage counter is updated
    by ``record_cost_usage()`` after the turn completes. Under concurrency,
    two parallel turns may both pass this check against the same snapshot.
    This is acceptable because cost-based limits are approximate by nature
    (the exact cost is unknown until after generation).

    Fails open: if Redis is unavailable, allows the request.
    """
    # Short-circuit: when both limits are 0 (unlimited) skip the Redis
    # round-trip entirely.
    if daily_cost_limit <= 0 and weekly_cost_limit <= 0:
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

    if daily_cost_limit > 0 and daily_used >= daily_cost_limit:
        raise RateLimitExceeded("daily", _daily_reset_time(now=now))

    if weekly_cost_limit > 0 and weekly_used >= weekly_cost_limit:
        raise RateLimitExceeded("weekly", _weekly_reset_time(now=now))


async def reset_daily_usage(user_id: str, daily_cost_limit: int = 0) -> bool:
    """Reset a user's daily cost usage counter in Redis.

    Called after a user pays credits to extend their daily limit.
    Also reduces the weekly usage counter by ``daily_cost_limit`` microdollars
    (clamped to 0) so the user effectively gets one extra day's worth of
    weekly capacity.

    Args:
        user_id: The user's ID.
        daily_cost_limit: The configured daily cost limit in microdollars.
            When positive, the weekly counter is reduced by this amount.

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
        w_key = _weekly_key(user_id, now=now) if daily_cost_limit > 0 else None

        pipe = redis.pipeline(transaction=True)
        pipe.delete(d_key)
        if w_key is not None:
            pipe.decrby(w_key, daily_cost_limit)
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


async def record_cost_usage(
    user_id: str,
    cost_microdollars: int,
) -> None:
    """Record a user's generation spend against daily and weekly counters.

    ``cost_microdollars`` is the real generation cost reported by the
    provider (OpenRouter's ``usage.cost`` or the Claude Agent SDK's
    ``total_cost_usd`` converted to microdollars). Because the provider
    cost already reflects model pricing and cache discounts, this function
    carries no pricing table or weighting — it just increments counters.

    Args:
        user_id: The user's ID.
        cost_microdollars: Spend to record in microdollars (1 USD = 1_000_000).
            Non-positive values are ignored.
    """
    cost_microdollars = max(0, cost_microdollars)
    if cost_microdollars <= 0:
        return

    logger.info("Recording copilot spend: %d microdollars", cost_microdollars)

    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        # Use MULTI/EXEC so each INCRBY/EXPIRE pair is atomic — guarantees
        # the TTL is set even if the connection drops mid-pipeline, so
        # counters can never survive past their date-based rotation window.
        pipe = redis.pipeline(transaction=True)

        # Daily counter (expires at next midnight UTC)
        d_key = _daily_key(user_id, now=now)
        pipe.incrby(d_key, cost_microdollars)
        seconds_until_daily_reset = int(
            (_daily_reset_time(now=now) - now).total_seconds()
        )
        pipe.expire(d_key, max(seconds_until_daily_reset, 1))

        # Weekly counter (expires end of week)
        w_key = _weekly_key(user_id, now=now)
        pipe.incrby(w_key, cost_microdollars)
        seconds_until_weekly_reset = int(
            (_weekly_reset_time(now=now) - now).total_seconds()
        )
        pipe.expire(w_key, max(seconds_until_weekly_reset, 1))

        await pipe.execute()
    except (RedisError, ConnectionError, OSError):
        logger.warning(
            "Redis unavailable for recording cost usage (microdollars=%d)",
            cost_microdollars,
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
    try:
        user = await user_db().get_user_by_id(user_id)
    except Exception:
        raise _UserNotFoundError(user_id)
    if user.subscription_tier:
        return SubscriptionTier(user.subscription_tier)
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

    Invalidates every cache that keys off the user's subscription tier so the
    change is visible immediately: this function's own ``get_user_tier``, the
    shared ``get_user_by_id`` (which exposes ``user.subscription_tier``), and
    ``get_pending_subscription_change`` (since an admin override can invalidate
    a cached ``cancel_at_period_end`` or schedule-based pending state).

    If the user has an active Stripe subscription whose current price does not
    match ``tier``, Stripe will keep billing the old price and the next
    ``customer.subscription.updated`` webhook will overwrite the DB tier back
    to whatever Stripe has. Proper reconciliation (cancelling or modifying the
    Stripe subscription when an admin overrides the tier) is out of scope for
    this PR — it changes the admin contract and needs its own test coverage.
    For now we emit a ``WARNING`` so drift surfaces via Sentry until that
    follow-up lands.

    Raises:
        prisma.errors.RecordNotFoundError: If the user does not exist.
    """
    await PrismaUser.prisma().update(
        where={"id": user_id},
        data={"subscriptionTier": tier.value},
    )
    get_user_tier.cache_delete(user_id)  # type: ignore[attr-defined]
    # Local import required: backend.data.credit imports backend.copilot.rate_limit
    # (via get_user_tier in credit.py's _invalidate_user_tier_caches), so a
    # top-level ``from backend.data.credit import ...`` here would create a
    # circular import at module-load time.
    from backend.data.credit import get_pending_subscription_change

    get_user_by_id.cache_delete(user_id)  # type: ignore[attr-defined]
    get_pending_subscription_change.cache_delete(user_id)  # type: ignore[attr-defined]

    # The DB write above is already committed; the drift check is best-effort
    # diagnostic logging. Fire-and-forget so admin bulk ops don't wait on a
    # Stripe roundtrip. The inner helper wraps its body in a timeout + broad
    # except so background task errors still surface via logs rather than as
    # "task exception never retrieved" warnings. Cancellation on request
    # shutdown is acceptable — the drift warning is non-load-bearing.
    asyncio.ensure_future(_drift_check_background(user_id, tier))


async def _drift_check_background(user_id: str, tier: SubscriptionTier) -> None:
    """Run the Stripe drift check in the background, logging rather than raising."""
    try:
        await asyncio.wait_for(
            _warn_if_stripe_subscription_drifts(user_id, tier),
            timeout=5.0,
        )
        logger.debug(
            "set_user_tier: drift check completed for user=%s admin_tier=%s",
            user_id,
            tier.value,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "set_user_tier: drift check timed out for user=%s admin_tier=%s",
            user_id,
            tier.value,
        )
    except asyncio.CancelledError:
        # Request may have completed and the event loop is cancelling tasks —
        # the drift log is non-critical, so accept cancellation silently.
        raise
    except Exception:
        logger.exception(
            "set_user_tier: drift check background task failed for"
            " user=%s admin_tier=%s",
            user_id,
            tier.value,
        )


async def _warn_if_stripe_subscription_drifts(
    user_id: str, new_tier: SubscriptionTier
) -> None:
    """Emit a WARNING when an admin tier override leaves an active Stripe sub on a
    mismatched price.

    The warning is diagnostic only: Stripe remains the billing source of truth,
    so the next ``customer.subscription.updated`` webhook will reset the DB
    tier. Surfacing the drift here lets ops catch admin overrides that bypass
    the intended Checkout / Portal cancel flows before users notice surprise
    charges.
    """
    # Local imports: see note in ``set_user_tier`` about the credit <-> rate_limit
    # circular. These helpers (``_get_active_subscription``,
    # ``get_subscription_price_id``) live in credit.py alongside the rest of
    # the Stripe billing code.
    from backend.data.credit import _get_active_subscription, get_subscription_price_id

    try:
        user = await get_user_by_id(user_id)
        if not getattr(user, "stripe_customer_id", None):
            return
        sub = await _get_active_subscription(user.stripe_customer_id)
        if sub is None:
            return
        items = sub["items"].data
        if not items:
            return
        price = items[0].price
        current_price_id = price if isinstance(price, str) else price.id
        # The LaunchDarkly-backed price lookup must live inside this try/except:
        # an LD SDK failure (network, token revoked) here would otherwise
        # propagate past set_user_tier's already-committed DB write and turn a
        # best-effort diagnostic into a 500 on admin tier writes.
        expected_price_id = await get_subscription_price_id(new_tier)
    except Exception:
        logger.debug(
            "_warn_if_stripe_subscription_drifts: drift lookup failed for"
            " user=%s; skipping drift warning",
            user_id,
            exc_info=True,
        )
        return
    if expected_price_id is not None and expected_price_id == current_price_id:
        return
    logger.warning(
        "Admin tier override will drift from Stripe: user=%s admin_tier=%s"
        " stripe_sub=%s stripe_price=%s expected_price=%s — the next"
        " customer.subscription.updated webhook will reconcile the DB tier"
        " back to whatever Stripe has; cancel or modify the Stripe subscription"
        " if you intended the admin override to stick.",
        user_id,
        new_tier.value,
        sub.id,
        current_price_id,
        expected_price_id,
    )


async def get_global_rate_limits(
    user_id: str,
    config_daily: int,
    config_weekly: int,
) -> tuple[int, int, SubscriptionTier]:
    """Resolve global rate limits from LaunchDarkly, falling back to config.

    Values are microdollars. The base limits (from LD or config) are
    multiplied by the user's tier multiplier so that higher tiers receive
    proportionally larger allowances.

    Args:
        user_id: User ID for LD flag evaluation context.
        config_daily: Fallback daily cost limit (microdollars) from ChatConfig.
        config_weekly: Fallback weekly cost limit (microdollars) from ChatConfig.

    Returns:
        (daily_cost_limit, weekly_cost_limit, tier) — limits in microdollars.
    """
    # Lazy import to avoid circular dependency:
    # rate_limit -> feature_flag -> settings -> ... -> rate_limit
    from backend.util.feature_flag import Flag, get_feature_flag_value

    # Fetch daily + weekly flags in parallel — each LD evaluation is an
    # independent network round-trip, so gather cuts latency roughly in half.
    daily_raw, weekly_raw = await asyncio.gather(
        get_feature_flag_value(
            Flag.COPILOT_DAILY_COST_LIMIT.value, user_id, config_daily
        ),
        get_feature_flag_value(
            Flag.COPILOT_WEEKLY_COST_LIMIT.value, user_id, config_weekly
        ),
    )
    try:
        daily = max(0, int(daily_raw))
    except (TypeError, ValueError):
        logger.warning("Invalid LD value for daily cost limit: %r", daily_raw)
        daily = config_daily
    try:
        weekly = max(0, int(weekly_raw))
    except (TypeError, ValueError):
        logger.warning("Invalid LD value for weekly cost limit: %r", weekly_raw)
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
