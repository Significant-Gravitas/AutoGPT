"""CoPilot rate limiting based on generation cost.

Uses Redis fixed-window counters to track per-user USD spend (stored as
microdollars, matching ``PlatformCostLog.cost_microdollars``) with
configurable daily and weekly limits. Daily windows reset at midnight UTC;
weekly windows reset at ISO week boundary (Monday 00:00 UTC). Fails open
when Redis is unavailable to avoid blocking users.

Storing microdollars rather than tokens means the counter already reflects
real model pricing (including cache discounts and provider surcharges), so
this module carries no pricing table — the cost comes from OpenRouter's
``usage.cost`` field (baseline), the Claude Agent SDK's reported total
cost (SDK path), web_search tool calls, and the prompt-simulation harness.

Boundary with the credit wallet
===============================

Microdollars (this module) and credits (``backend.data.block_cost_config``)
are intentionally separate budgets:

* **Credits** are the user-facing prepaid wallet. Every block invocation
  that has a ``BlockCost`` entry decrements credits — this is what the
  user buys, tops up, and sees on the billing page.  Marketplace blocks
  may also charge credits to block creators. The credit charge is a flat
  per-run amount sourced from ``BLOCK_COSTS``.  Copilot ``run_block``
  calls go through this path too: block execution bills the user's
  credit wallet, not this counter.
* **Microdollars** meter AutoGPT's **operator-side infrastructure cost**
  for the copilot **LLM turn itself** — the real USD we spend on the
  baseline model, Claude Agent SDK runs, the web_search tool, and the
  prompt simulator. They gate the chat loop so a single user can't burn
  the daily / weekly infra budget driving the chat regardless of their
  credit balance. BYOK runs (user supplied their own API key) do **not**
  decrement this counter — the user is paying the provider, not us.

A future option is to unify these into one wallet; until then the
boundary above is the contract.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from enum import Enum

from prisma.models import User as PrismaUser
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from backend.data.db_accessors import user_db
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.data.user import get_user_by_id
from backend.util.cache import cached

logger = logging.getLogger(__name__)

# "copilot:cost" (not the legacy "copilot:usage") so stale token-based
# counters are not misread as microdollars.
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

    NO_TIER = "NO_TIER"
    BASIC = "BASIC"
    PRO = "PRO"
    MAX = "MAX"
    BUSINESS = "BUSINESS"
    ENTERPRISE = "ENTERPRISE"


# Default multiplier applied to the base cost limits (from LD / config) for each
# tier. Used as the fallback when the LD flag ``copilot-tier-multipliers`` is
# unset or unparseable — see ``get_tier_multipliers``.  BUSINESS matches
# ENTERPRISE (60x); MAX sits at 20x as the self-service $320 tier. Float-typed
# so LD-provided fractional multipliers (e.g. 8.5×) compose naturally; the
# eventual ``int(base * multiplier)`` in ``get_global_rate_limits`` keeps the
# downstream microdollar math integer.
_DEFAULT_TIER_MULTIPLIERS: dict[SubscriptionTier, float] = {
    # NO_TIER is the explicit "no active Stripe subscription" state —
    # multiplier 0.0 collapses the per-period limit to int(base * 0) = 0, so
    # all rate-limited routes (CoPilot chat, AutoPilot) refuse with 429
    # before any business logic runs. This is the backend half of the
    # paywall (the frontend modal nudges UI users; this gate enforces
    # server-side regardless of client). BASIC stays as a future paid-tier
    # option; for now it falls back to the same baseline as paid tiers.
    SubscriptionTier.NO_TIER: 0.0,
    SubscriptionTier.BASIC: 1.0,
    SubscriptionTier.PRO: 5.0,
    SubscriptionTier.MAX: 20.0,
    SubscriptionTier.BUSINESS: 60.0,
    SubscriptionTier.ENTERPRISE: 60.0,
}

# Public re-export retained for backward compatibility with call-sites / tests
# that historically read ``TIER_MULTIPLIERS`` directly.  New code should prefer
# ``get_tier_multipliers`` so LD overrides are honoured.
TIER_MULTIPLIERS = _DEFAULT_TIER_MULTIPLIERS

DEFAULT_TIER = SubscriptionTier.NO_TIER


@cached(ttl_seconds=60, maxsize=1, cache_none=False)
async def _fetch_tier_multipliers_flag() -> dict[SubscriptionTier, float] | None:
    """Fetch the ``copilot-tier-multipliers`` LD flag and parse it.

    Returns a sparse ``{tier: multiplier}`` map built from whichever keys are
    valid in the flag payload, or ``None`` when the flag is unset / invalid /
    LD is unavailable.  ``cache_none=False`` avoids pinning a transient LD
    failure for a full minute — the next call retries.

    The LD value is expected to be a JSON object keyed by tier enum name
    (``{"BASIC": 1, "PRO": 5, "BUSINESS": 20.5}``).  Unknown tier keys and
    non-numeric / non-positive values are skipped; callers merge whatever
    survives into :data:`_DEFAULT_TIER_MULTIPLIERS`.
    """
    # Lazy import: rate_limit -> feature_flag -> settings -> ... -> rate_limit.
    from backend.util.feature_flag import Flag, get_feature_flag_value

    raw = await get_feature_flag_value(
        Flag.COPILOT_TIER_MULTIPLIERS.value, "system", None
    )
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Invalid LD value for copilot-tier-multipliers (expected JSON object): %r",
            raw,
        )
        return None

    parsed: dict[SubscriptionTier, float] = {}
    for key, value in raw.items():
        try:
            tier = SubscriptionTier(key)
        except ValueError:
            continue
        try:
            multiplier = float(value)
        except (TypeError, ValueError):
            continue
        if multiplier <= 0:
            continue
        parsed[tier] = multiplier
    return parsed or None


@cached(ttl_seconds=60, maxsize=1, cache_none=False)
async def _fetch_cost_limits_flag() -> dict[str, int] | None:
    """Fetch the ``copilot-cost-limits`` LD flag and parse it.

    Returns a sparse ``{"daily": int, "weekly": int}`` map built from whichever
    keys are valid in the flag payload, or ``None`` when the flag is unset /
    invalid / LD is unavailable.  Callers merge whatever survives into their
    config defaults (see :func:`get_global_rate_limits`).

    The LD value is expected to be a JSON object keyed by window name
    (``{"daily": 625000, "weekly": 3125000}``).  Non-int / negative values
    are skipped so a broken key degrades to the config default instead of
    wiping out the limit.
    """
    # Lazy import: rate_limit -> feature_flag -> settings -> ... -> rate_limit.
    from backend.util.feature_flag import Flag, get_feature_flag_value

    raw = await get_feature_flag_value(Flag.COPILOT_COST_LIMITS.value, "system", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Invalid LD value for copilot-cost-limits (expected JSON object): %r",
            raw,
        )
        return None

    parsed: dict[str, int] = {}
    for key in ("daily", "weekly"):
        if key not in raw:
            continue
        value = raw[key]
        # Strict int check — booleans are subclasses of int in Python, and we
        # don't want to coerce strings like "100" or floats like 1.9 silently
        # into a rate-limit cap. Docstring says "non-int values are skipped".
        if isinstance(value, bool) or not isinstance(value, int):
            logger.warning(
                "Invalid LD value for copilot-cost-limits[%s]: %r", key, value
            )
            continue
        if value < 0:
            logger.warning(
                "Negative LD value for copilot-cost-limits[%s]: %r", key, value
            )
            continue
        parsed[key] = value
    return parsed or None


async def get_tier_multipliers() -> dict[str, float]:
    """Return the effective ``{tier_value: multiplier}`` map.

    Honours the ``copilot-tier-multipliers`` LD flag when set; missing tiers
    inherit :data:`_DEFAULT_TIER_MULTIPLIERS`.  Unparseable flag values or LD
    fetch failures fall back to the defaults without raising.

    Keys are the tier enum string values (``"BASIC"``, ``"PRO"``, …) rather
    than the enum itself so callers holding ``prisma.enums.SubscriptionTier``
    don't hit a spurious mismatch against this module's local mirror.

    The flag is evaluated system-wide — per-tier multipliers are a global knob.
    If per-cohort overrides are ever needed, add a user_id parameter here and
    thread it through ``_fetch_tier_multipliers_flag`` to LD.
    """
    try:
        override = await _fetch_tier_multipliers_flag()
    except Exception:
        # LD SDK / Redis / network failures here are best-effort — fall back.
        logger.warning("get_tier_multipliers: LD lookup failed", exc_info=True)
        override = None
    merged: dict[SubscriptionTier, float] = dict(_DEFAULT_TIER_MULTIPLIERS)
    if override:
        merged.update(override)
    return {tier.value: multiplier for tier, multiplier in merged.items()}


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

        d_key = _daily_key(user_id, now=now)
        w_key = _weekly_key(user_id, now=now) if daily_cost_limit > 0 else None

        # Daily and weekly keys hash to different cluster slots, so cross-key
        # MULTI/EXEC is not available. Issue the writes sequentially — the
        # failure mode (daily deleted, weekly not decremented) is a
        # best-effort refund budget that the read path already tolerates.
        await redis.delete(d_key)
        if w_key is not None:
            await _decr_counter_floor_zero(redis, w_key, daily_cost_limit)

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
    d_key = _daily_key(user_id, now=now)
    w_key = _weekly_key(user_id, now=now)
    daily_ttl = max(int((_daily_reset_time(now=now) - now).total_seconds()), 1)
    weekly_ttl = max(int((_weekly_reset_time(now=now) - now).total_seconds()), 1)
    try:
        redis = await get_redis_async()
        # Daily and weekly keys hash to different cluster slots — cross-slot
        # MULTI/EXEC is not supported, so each counter gets its own
        # single-key transaction. Per-counter INCRBY+EXPIRE atomicity is the
        # invariant that matters; the two counters are independent budgets.
        await _incr_counter_atomic(redis, d_key, cost_microdollars, daily_ttl)
        await _incr_counter_atomic(redis, w_key, cost_microdollars, weekly_ttl)
    except (RedisError, ConnectionError, OSError):
        logger.warning(
            "Redis unavailable for recording cost usage (microdollars=%d)",
            cost_microdollars,
        )


async def _incr_counter_atomic(
    redis: AsyncRedisClient, key: str, delta: int, ttl_seconds: int
) -> None:
    """INCRBY + EXPIRE on a single key inside a MULTI/EXEC transaction."""
    pipe = redis.pipeline(transaction=True)
    pipe.incrby(key, delta)
    pipe.expire(key, ttl_seconds)
    await pipe.execute()


# Atomic DECRBY + floor-to-zero so a concurrent INCRBY from record_cost_usage
# cannot be lost. DELETE on underflow also avoids leaving a zero-valued key
# with no TTL, which the non-atomic set-with-keepttl variant could do.
_DECR_FLOOR_ZERO_SCRIPT = """
local value = redis.call("DECRBY", KEYS[1], ARGV[1])
if value < 0 then
    redis.call("DEL", KEYS[1])
    return 0
end
return value
"""


async def _decr_counter_floor_zero(
    redis: AsyncRedisClient, key: str, delta: int
) -> None:
    """Atomically DECRBY ``delta`` on ``key`` and DEL on underflow.

    DEL on underflow avoids leaving a zero-valued key without a TTL, so the
    next INCRBY in ``record_cost_usage`` re-seeds both the value and the
    expiry in one shot.
    """
    await redis.eval(_DECR_FLOOR_ZERO_SCRIPT, 1, key, delta)


class _UserNotFoundError(Exception):
    """Raised when a user record is missing or has no subscription tier.

    Raising (rather than returning ``DEFAULT_TIER``) prevents ``@cached``
    from persisting the fallback, which would otherwise keep serving FREE
    for up to the TTL after the user's real tier is set.
    """


@cached(maxsize=1000, ttl_seconds=300, shared_cache=True)
async def _fetch_user_tier(user_id: str) -> SubscriptionTier:
    """Fetch the user's rate-limit tier, cached across pods.

    Only successful lookups are cached. Missing users raise
    ``_UserNotFoundError`` so ``@cached`` never stores the fallback.
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

    Invalidates the caches that expose ``subscription_tier`` so the change
    takes effect immediately. If the user has an active Stripe subscription
    on a mismatched price, emits a WARNING; Stripe remains the billing
    source of truth and the next webhook will reconcile the DB tier.

    Raises:
        prisma.errors.RecordNotFoundError: If the user does not exist.
    """
    await PrismaUser.prisma().update(
        where={"id": user_id},
        data={"subscriptionTier": tier.value},
    )
    get_user_tier.cache_delete(user_id)  # type: ignore[attr-defined]
    # Local import: backend.data.credit imports from this module.
    from backend.data.credit import get_pending_subscription_change

    get_user_by_id.cache_delete(user_id)  # type: ignore[attr-defined]
    get_pending_subscription_change.cache_delete(user_id)  # type: ignore[attr-defined]

    # Fire-and-forget drift check so admin bulk ops don't wait on Stripe.
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
    """Emit a WARNING when an admin tier override leaves an active Stripe
    subscription on a mismatched price."""
    # Local import: breaks a credit <-> rate_limit circular at module load.
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
        # Inside the try/except: an LD SDK failure here must not turn a
        # best-effort diagnostic into a 500 after the DB write committed.
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
    try:
        override = await _fetch_cost_limits_flag()
    except Exception:
        logger.warning("get_global_rate_limits: LD lookup failed", exc_info=True)
        override = None
    override = override or {}
    daily = override.get("daily", config_daily)
    weekly = override.get("weekly", config_weekly)

    # Apply tier multiplier — resolved through LD (copilot-tier-multipliers)
    # so multipliers can be tuned without a deploy. Falls back to the defaults
    # when LD is unavailable.
    tier = await get_user_tier(user_id)
    multipliers = await get_tier_multipliers()
    multiplier = multipliers.get(tier.value, 1.0)
    # NO_TIER's 0.0 multiplier is the backend half of the paywall — it
    # collapses limits to zero so unsubscribed users can't run the chat.
    # Only enforce that gate when the platform-payment flag is on for this
    # user; in the beta cohort (flag off) NO_TIER falls back to BASIC's
    # baseline so the e2e suite and beta testers retain access.
    if tier == SubscriptionTier.NO_TIER:
        from backend.util.feature_flag import Flag, is_feature_enabled

        if not await is_feature_enabled(Flag.ENABLE_PLATFORM_PAYMENT, user_id):
            multiplier = multipliers.get(SubscriptionTier.BASIC.value, 1.0)
    if multiplier != 1.0:
        # Cast back to int to preserve the microdollar integer contract
        # downstream — fractional LD multipliers (e.g. 8.5×) truncate at the
        # last microdollar, which is well below any meaningful precision.
        daily = int(daily * multiplier)
        weekly = int(weekly * multiplier)

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
    d_key = _daily_key(user_id, now=now)
    w_key = _weekly_key(user_id, now=now) if reset_weekly else None
    try:
        redis = await get_redis_async()
        # Daily and weekly keys hash to different cluster slots — multi-key
        # DELETE would raise CROSSSLOT, so issue separate calls.
        await redis.delete(d_key)
        if w_key is not None:
            await redis.delete(w_key)
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
