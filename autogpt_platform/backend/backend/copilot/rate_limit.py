"""CoPilot rate limiting based on generation cost.

Uses Redis fixed-window counters to track per-user USD spend (stored as
microdollars, matching ``PlatformCostLog.cost_microdollars``) with
configurable daily and weekly limits. Daily windows reset at midnight UTC;
weekly windows reset at ISO week boundary (Monday 00:00 UTC).

Failure-mode policy:

* Enforcement path (:func:`check_rate_limit`) **fails closed** — if Redis
  is unreachable we raise :class:`RateLimitUnavailable` so the API layer
  returns HTTP 503. A brown-out must not let a user bypass their
  daily / weekly USD cap.
* Observability paths (:func:`get_usage_status`, the reset-count
  read/write helpers, the recording path :func:`record_cost_usage`) keep
  fail-open / best-effort semantics — losing a usage gauge or a single
  cost increment is preferable to 500-ing the request, and the
  authoritative cap is re-checked on the next turn.
* Reset paths (:func:`reset_user_usage`, :func:`reset_daily_usage`,
  :func:`acquire_reset_lock`) re-raise / return ``False`` so billed reset
  operations cannot silently no-op.

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

import fastapi
from autogpt_libs.auth.dependencies import get_user_id
from prisma.models import User as PrismaUser
from pydantic import BaseModel, Field
from redis.exceptions import RedisClusterException, RedisError

from backend.data.db_accessors import user_db
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.data.user import get_user_by_id
from backend.util.cache import cached
from backend.util.feature_flag import Flag, get_feature_flag_value, is_feature_enabled

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


# Per-tier workspace storage caps in MB. NO_TIER keeps the same baseline as
# BASIC so users who cancel retain a small quota and see a real overage cap,
# while LaunchDarkly can still tune tiers without a deploy.
_DEFAULT_TIER_WORKSPACE_STORAGE_MB: dict[SubscriptionTier, int] = {
    SubscriptionTier.NO_TIER: 250,  # 250 MB
    SubscriptionTier.BASIC: 250,  # 250 MB
    SubscriptionTier.PRO: 1024,  # 1 GB
    SubscriptionTier.MAX: 5 * 1024,  # 5 GB
    SubscriptionTier.BUSINESS: 15 * 1024,  # 15 GB
    SubscriptionTier.ENTERPRISE: 15 * 1024,  # 15 GB
}


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


@cached(ttl_seconds=60, maxsize=1, cache_none=False)
async def _fetch_workspace_storage_limits_flag() -> dict[SubscriptionTier, int] | None:
    """Fetch the ``copilot-tier-workspace-storage-limits`` LD flag and parse it.

    Returns a sparse ``{tier: megabytes}`` map built from whichever keys are
    valid in the flag payload, or ``None`` when the flag is unset / invalid /
    LD is unavailable. Callers merge whatever survives into
    :data:`_DEFAULT_TIER_WORKSPACE_STORAGE_MB`.

    The LD value is expected to be a JSON object keyed by tier enum name
    (``{"NO_TIER": 250, "PRO": 1024, "BUSINESS": 15360}``). Non-int or
    negative values are skipped so a broken key degrades to the code default
    instead of wiping out the limit.
    """
    raw = await get_feature_flag_value(
        Flag.COPILOT_TIER_WORKSPACE_STORAGE_LIMITS.value, "system", None
    )
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Invalid LD value for copilot-tier-workspace-storage-limits "
            "(expected JSON object): %r",
            raw,
        )
        return None

    parsed: dict[SubscriptionTier, int] = {}
    for key, value in raw.items():
        try:
            tier = SubscriptionTier(key)
        except ValueError:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            logger.warning(
                "Invalid LD value for copilot-tier-workspace-storage-limits[%s]: %r",
                key,
                value,
            )
            continue
        if value <= 0:
            logger.warning(
                "Non-positive LD value for copilot-tier-workspace-storage-limits[%s]: %r",
                key,
                value,
            )
            continue
        parsed[tier] = value
    return parsed or None


async def get_workspace_storage_limits_mb() -> dict[str, int]:
    """Return the effective ``{tier_value: megabytes}`` workspace limit map.

    Honours the ``copilot-tier-workspace-storage-limits`` LD flag when set;
    missing tiers inherit :data:`_DEFAULT_TIER_WORKSPACE_STORAGE_MB`.
    Unparseable flag values or LD fetch failures fall back to the defaults.
    """
    try:
        override = await _fetch_workspace_storage_limits_flag()
    except Exception:
        logger.warning(
            "get_workspace_storage_limits_mb: LD lookup failed", exc_info=True
        )
        override = None

    merged: dict[SubscriptionTier, int] = dict(_DEFAULT_TIER_WORKSPACE_STORAGE_MB)
    if override:
        merged.update(override)
    return {tier.value: megabytes for tier, megabytes in merged.items()}


class UsageWindow(BaseModel):
    """Usage within a single time window.

    ``used`` and ``limit`` are in microdollars (1 USD = 1_000_000).
    """

    used: int
    limit: int = Field(
        description="Maximum microdollars of spend allowed in this window. "
        "0 means no spend allowed (the user is over-cap immediately); there "
        "is no unlimited tier — the public model uses ``None`` for "
        "no-cap-configured."
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
            if w.limit < 0:
                # Defensive: nothing produces a negative limit today, but
                # treat it as "no cap configured" → hide the window from
                # the UI rather than divide-by-negative.
                return None
            if w.limit == 0:
                # Limit of 0 means "no spend allowed" — surface as fully
                # exhausted so the UI shows the user as blocked instead
                # of silently treating it as null/unlimited (which was
                # the source of the original autopilot paywall bypass).
                return UsageWindowPublic(
                    percent_used=100.0,
                    resets_at=w.resets_at,
                )
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


class RateLimitUnavailable(Exception):
    """Rate limit state is currently unreachable — request rejected to
    prevent USD-cap bypass. Maps to HTTP 503 in the API layer.

    Distinct from :class:`RateLimitExceeded` (HTTP 429): the user is not
    over their cap, but Redis is down so we cannot prove they are under it
    either. Failing closed avoids the brown-out bypass where a user could
    blast the LLM during a Redis outage and exceed their daily/weekly USD
    allowance by hundreds of dollars.
    """


class UserPaywalledError(Exception):
    """User has no entitlement to run a paywalled feature (NO_TIER tier
    + ``ENABLE_PLATFORM_PAYMENT`` on).

    Raised by ``add_graph_execution`` and other deep enqueue paths so
    that *every* execution entry point (HTTP route, scheduled cron,
    webhook trigger, external API, internal copilot tool) gets the same
    gate without each one having to remember a route-level dependency.
    Routes wrap this into HTTP 402; background tasks log and abandon
    the run.
    """

    def __init__(
        self,
        message: str = "A subscription is required to run this feature.",
    ) -> None:
        super().__init__(message)


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
        daily_cost_limit: Max microdollars of spend per day. 0 means no spend
            allowed (over-cap immediately).
        weekly_cost_limit: Max microdollars of spend per week. 0 means no
            spend allowed (over-cap immediately).
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
    except (RedisError, RedisClusterException, ConnectionError, OSError, ValueError):
        # ValueError: corrupt non-numeric counter (partial write / wrong-type
        # SET) — same fail-open semantics, returns zeros.
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


async def get_remaining_usd_budget(
    user_id: str,
    daily_cost_limit: int,
    weekly_cost_limit: int,
    floor_usd: float = 0.5,
) -> float:
    """Return the user's remaining USD spend cap for the current windows.

    The result is the smaller of ``daily_remaining`` and ``weekly_remaining``
    expressed in USD.  Used to size the SDK's per-query ``max_budget_usd``
    so the in-CLI "wrap up gracefully" reminder fires earlier when the
    user is close to their actual cap, and to feed the baseline path's
    per-turn budget hint via :func:`build_budget_ctx`.

    A limit of ``0`` is treated as "no spend allowed" — remaining = 0
    on that window. There is no real-world unlimited tier; callers
    should not pass 0 expecting it to mean "no cap".

    Failure modes:
        * Redis brown-out → ``floor_usd`` (so callers using the value
          as a soft hint don't pretend the user has full budget; the
          pre-turn gate has already failed closed at 503 in this case,
          so we only land here from observability paths).

    Args:
        user_id: The user's ID.
        daily_cost_limit: Daily cap in microdollars. 0 = no spend allowed
            on this window.
        weekly_cost_limit: Weekly cap in microdollars. 0 = no spend allowed
            on this window.
        floor_usd: Lower bound on the returned value (USD).  Avoids
            handing the SDK a degenerate ``$0`` budget that would refuse
            to start a turn.  Set to ``0.0`` when the caller wants a
            faithful "no remaining budget" signal instead of a floor.
    """
    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        daily_raw, weekly_raw = await asyncio.gather(
            redis.get(_daily_key(user_id, now=now)),
            redis.get(_weekly_key(user_id, now=now)),
        )
        daily_used = int(daily_raw or 0)
        weekly_used = int(weekly_raw or 0)
    except (RedisError, RedisClusterException, ConnectionError, OSError, ValueError):
        logger.warning("Redis unavailable for remaining-budget lookup, returning floor")
        return floor_usd

    # ``>= 0`` (not ``> 0``): a limit of 0 is "no spend allowed", so the
    # remaining is 0 on that window. Mirrors check_rate_limit's semantics
    # — there is no unlimited tier, so we never short-circuit to float(inf).
    remaining_microdollars = float("inf")
    if daily_cost_limit >= 0:
        remaining_microdollars = min(
            remaining_microdollars, max(0, daily_cost_limit - daily_used)
        )
    if weekly_cost_limit >= 0:
        remaining_microdollars = min(
            remaining_microdollars, max(0, weekly_cost_limit - weekly_used)
        )
    remaining_usd = (
        remaining_microdollars / 1_000_000.0
        if remaining_microdollars != float("inf")
        else float("inf")
    )
    return max(floor_usd, remaining_usd)


async def build_budget_ctx(
    user_id: str | None,
    default_daily_cost_limit: int,
    default_weekly_cost_limit: int,
) -> str:
    """Build the inner content for an ``<budget_context>`` block.

    Returns the *inner* text — the caller (``inject_user_context``)
    wraps it in the ``<budget_context>`` tag.  Combines the tier-limit
    lookup (``get_global_rate_limits``) and the remaining-USD lookup
    (``get_remaining_usd_budget``) so callers don't have to compose
    them by hand on every turn.

    Returns ``""`` when:
        * no ``user_id`` is available,
        * the user has 0 remaining budget (they shouldn't be here — the
          paywall dep raises 402 first — but defend against drift),
        * Redis is unavailable (we'd rather emit nothing than a
          misleading ``$0.00`` hint — the pre-turn gate already fails
          closed at 503 in that case).
    """
    if not user_id:
        return ""
    daily_limit, weekly_limit, _tier = await get_global_rate_limits(
        user_id,
        default_daily_cost_limit,
        default_weekly_cost_limit,
    )
    remaining = await get_remaining_usd_budget(
        user_id=user_id,
        daily_cost_limit=daily_limit,
        weekly_cost_limit=weekly_limit,
        # 0.0 here is a sentinel meaning "Redis brown-out / no value" —
        # we map it back to "" below so the model doesn't see a
        # misleading $0.00 hint when our metrics are degraded.
        floor_usd=0.0,
    )
    if remaining == float("inf") or remaining <= 0.0:
        return ""
    return (
        f"Approximate remaining USD budget for this user: ${remaining:.2f}.\n"
        "Pace your tool use and reasoning depth so the response stays "
        "within this envelope."
    )


async def check_rate_limit(
    user_id: str,
    daily_cost_limit: int,
    weekly_cost_limit: int,
) -> None:
    """Check if user is within rate limits.

    Raises :class:`RateLimitExceeded` when the user is at-or-over their cap
    and :class:`RateLimitUnavailable` when Redis is unreachable so the
    caller must fail closed (HTTP 503) — the daily/weekly USD caps are
    real money and cannot be bypassed by a Redis brown-out.

    A limit of ``0`` means "no spend allowed", not "unlimited" — there is
    no real-world unlimited tier. Routes that want to skip rate-limiting
    entirely should not call this function. Entitlement (``NO_TIER`` +
    ``ENABLE_PLATFORM_PAYMENT``) is enforced upstream by the route
    dependency :func:`enforce_payment_paywall`, so this function is
    purely about per-window USD usage.

    This is a pre-turn soft check. The authoritative usage counter is updated
    by ``record_cost_usage()`` after the turn completes. Under concurrency,
    two parallel turns may both pass this check against the same snapshot.
    This is acceptable because cost-based limits are approximate by nature
    (the exact cost is unknown until after generation).
    """
    now = datetime.now(UTC)
    try:
        redis = await get_redis_async()
        daily_raw, weekly_raw = await asyncio.gather(
            redis.get(_daily_key(user_id, now=now)),
            redis.get(_weekly_key(user_id, now=now)),
        )
        daily_used = int(daily_raw or 0)
        weekly_used = int(weekly_raw or 0)
    except (
        RedisError,
        RedisClusterException,
        ConnectionError,
        OSError,
        ValueError,
    ) as exc:
        # RedisClusterException covers SlotNotCoveredError raised during a
        # GKE rolling restart (it does NOT inherit from RedisError, only
        # from Exception, so it would otherwise bubble up as a 500 — which
        # is exactly the brown-out scenario this PR is meant to handle).
        # ValueError covers a corrupt non-numeric counter value (partial
        # write or wrong-type SET). Same spirit either way: we cannot prove
        # the user is under their cap, so fail closed.
        logger.warning("Rate limit state unreadable, rejecting request: %s", exc)
        raise RateLimitUnavailable() from exc

    # ``>= 0`` (not ``> 0``): a limit of 0 means "no spend allowed", and
    # any usage at or above 0 is over-cap. The previous ``> 0`` check
    # silently treated 0 as unlimited, which collided with the multiplier-
    # collapse semantics of :func:`get_global_rate_limits` and produced
    # the autopilot paywall bypass.
    if daily_cost_limit >= 0 and daily_used >= daily_cost_limit:
        raise RateLimitExceeded("daily", _daily_reset_time(now=now))

    if weekly_cost_limit >= 0 and weekly_used >= weekly_cost_limit:
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
    except (RedisError, RedisClusterException, ConnectionError, OSError):
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
    except (RedisError, RedisClusterException, ConnectionError, OSError) as exc:
        logger.warning("Redis unavailable for reset lock, rejecting reset: %s", exc)
        return False


async def release_reset_lock(user_id: str) -> None:
    """Release the per-user reset lock."""
    try:
        redis = await get_redis_async()
        await redis.delete(f"{_RESET_LOCK_PREFIX}:{user_id}")
    except (RedisError, RedisClusterException, ConnectionError, OSError):
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
    except (RedisError, RedisClusterException, ConnectionError, OSError, ValueError):
        # ValueError: corrupt non-numeric counter — fail-closed for billed
        # resets (returning None makes the caller refuse the billed reset).
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
    except (RedisError, RedisClusterException, ConnectionError, OSError):
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
    except (RedisError, RedisClusterException, ConnectionError, OSError):
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

    Distinguishes "row genuinely missing" (``user_db().get_user_by_id``
    raises ``ValueError``) from "transient DB / Prisma error" (other
    exceptions propagate as-is). Without this distinction a Supabase
    blip would silently degrade every paying user to NO_TIER and
    ``enforce_payment_paywall`` would 402 them — contradicting its
    503-on-blip contract.
    """
    try:
        user = await user_db().get_user_by_id(user_id)
    except ValueError:
        # ValueError = "User not found" per get_user_by_id's contract.
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


async def get_workspace_storage_limit_bytes(user_id: str) -> int:
    """Return the workspace storage cap in bytes for the user's subscription tier."""
    tier = await get_user_tier(user_id)
    limits_mb = await get_workspace_storage_limits_mb()
    tier_key = getattr(tier, "value", str(tier))
    fallback_mb = limits_mb.get(
        DEFAULT_TIER.value,
        _DEFAULT_TIER_WORKSPACE_STORAGE_MB[DEFAULT_TIER],
    )
    mb = limits_mb.get(tier_key, fallback_mb)
    return mb * 1024 * 1024


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
        # Match either cycle so a yearly subscriber on the right tier doesn't
        # spuriously trigger the drift warning.
        expected_monthly, expected_yearly = await asyncio.gather(
            get_subscription_price_id(new_tier, "monthly"),
            get_subscription_price_id(new_tier, "yearly"),
        )
    except Exception:
        logger.debug(
            "_warn_if_stripe_subscription_drifts: drift lookup failed for"
            " user=%s; skipping drift warning",
            user_id,
            exc_info=True,
        )
        return
    if current_price_id and current_price_id in (expected_monthly, expected_yearly):
        return
    logger.warning(
        "Admin tier override will drift from Stripe: user=%s admin_tier=%s"
        " stripe_sub=%s stripe_price=%s expected_prices=(monthly=%s, yearly=%s)"
        " — the next customer.subscription.updated webhook will reconcile the"
        " DB tier back to whatever Stripe has; cancel or modify the Stripe"
        " subscription if you intended the admin override to stick.",
        user_id,
        new_tier.value,
        sub.id,
        current_price_id,
        expected_monthly,
        expected_yearly,
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
    # when LD is unavailable. NO_TIER+paywall is gated upstream by
    # ``enforce_payment_paywall`` (HTTP 402 before this is reached); the
    # only NO_TIER path that gets here is the beta cohort (flag off), which
    # falls back to BASIC limits so testers retain access.
    tier = await get_user_tier(user_id)
    multipliers = await get_tier_multipliers()
    multiplier = multipliers.get(tier.value, 1.0)
    if tier == SubscriptionTier.NO_TIER and not await is_feature_enabled(
        Flag.ENABLE_PLATFORM_PAYMENT, user_id
    ):
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
    except (RedisError, RedisClusterException, ConnectionError, OSError):
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


# ---------------------------------------------------------------------------
# Paywall enforcement
# ---------------------------------------------------------------------------
#
# Two primitives. Every gate uses one or the other:
#
#   1. ``is_user_paywalled(user_id) -> bool``  — the check. Lookup errors
#      propagate so callers decide their own failure-mode.
#   2. ``enforce_payment_paywall(user_id)``    — HTTP gate. Wraps (1) and
#      raises ``UserPaywalledError`` (handler → 402) when paywalled, or
#      ``HTTPException(503)`` + Retry-After on lookup error. Doubles as
#      JWT route dep AND inline call from non-JWT routes (the explicit
#      ``user_id`` arg overrides the ``Security`` default).
#
# Background callers (scheduled cron, webhook handlers, copilot internal
# tools, ``add_graph_execution``) skip (2) and use (1) directly + raise
# ``UserPaywalledError`` inline, so the framework's own retry layer
# catches lookup errors — synthesising an ``HTTPException`` would be
# wrong shape for non-HTTP contexts.
#


async def is_user_paywalled(user_id: str) -> bool:
    """Return ``True`` if the user has no entitlement to paywalled features.

    A user with no DB tier record (``_UserNotFoundError`` — fresh signup
    that hasn't been provisioned yet, or row missing) is treated as
    ``NO_TIER`` here: paywalled iff ``ENABLE_PLATFORM_PAYMENT`` is on.
    Without this branch the missing-tier case would propagate as a 500
    in any caller that doesn't already have a generic ``except`` (e.g.
    the external API ``execute_graph_block`` route).

    Other tier-lookup errors propagate — callers decide (route → 503,
    background job → fail-open).
    """
    try:
        tier = await _fetch_user_tier(user_id)
    except _UserNotFoundError:
        # No DB row / no subscription_tier set — fresh signup that hasn't
        # been provisioned yet, or row missing entirely. Logged at debug
        # so ops can correlate "402s on fresh signups" with provisioning
        # gaps without spamming the warning level.
        logger.debug(
            "is_user_paywalled: tier lookup empty for %s, treating as NO_TIER",
            user_id[:8],
        )
        tier = SubscriptionTier.NO_TIER
    if tier != SubscriptionTier.NO_TIER:
        return False
    return await is_feature_enabled(Flag.ENABLE_PLATFORM_PAYMENT, user_id)


async def enforce_payment_paywall(
    user_id: str = fastapi.Security(get_user_id),
) -> None:
    """HTTP paywall gate — fail-closed on lookup error.

    Two call shapes, same behaviour:

    1. **JWT route dep**:
       ``dependencies=[Depends(enforce_payment_paywall)]`` — FastAPI
       auto-fills ``user_id`` from the JWT via the ``Security``
       default. Mirrors the ``requires_admin_user`` pattern.

    2. **Non-JWT route (inline)**: ``await enforce_payment_paywall(
       auth.user_id)`` — for API-key-auth routes (external API) where
       the JWT-based ``Security`` default doesn't apply; the explicit
       positional argument wins.

    Raises :class:`UserPaywalledError` (handled by the app-level
    exception handler → HTTP 402) if the user is on NO_TIER with
    ``ENABLE_PLATFORM_PAYMENT`` on. Tier-lookup failures map to
    **HTTP 503 + Retry-After** so the client retries instead of
    treating a transient Supabase / LD blip as a permanent paywall.

    Background callers (scheduled cron, webhook handlers, copilot
    internal tools) skip this function and call :func:`is_user_paywalled`
    directly + raise inline, so the background framework's own error
    handling decides what to do on lookup failure (typically: retry
    on next tick) instead of synthesising an HTTP-shaped exception.
    """
    try:
        paywalled = await is_user_paywalled(user_id)
    except Exception as exc:
        logger.warning(
            "enforce_payment_paywall: tier lookup failed for %s: %s",
            user_id[:8],
            exc,
        )
        raise fastapi.HTTPException(
            status_code=503,
            detail="Subscription state temporarily unavailable, retry shortly.",
            headers={"Retry-After": "30"},
        ) from exc
    if paywalled:
        raise UserPaywalledError(
            "A subscription is required to use this feature. Upgrade to continue."
        )
