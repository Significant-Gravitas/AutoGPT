"""Authoritative subscription-tier lookup and ordering policy."""

import logging

from prisma.enums import SubscriptionTier

from backend.util.cache import cached
from backend.util.clients import get_database_manager_async_client
from backend.util.service import UnhealthyServiceError

logger = logging.getLogger(__name__)

# Ordered from least to most privileged for minimum-tier checks.
SUBSCRIPTION_TIER_ORDER = (
    SubscriptionTier.NO_TIER,
    SubscriptionTier.BASIC,
    SubscriptionTier.PRO,
    SubscriptionTier.MAX,
    SubscriptionTier.BUSINESS,
    SubscriptionTier.ENTERPRISE,
)
_SUBSCRIPTION_TIER_RANK = {
    tier: rank for rank, tier in enumerate(SUBSCRIPTION_TIER_ORDER)
}


class SubscriptionTierUserNotFoundError(Exception):
    """Raised when Database Manager confirms that the user does not exist."""


@cached(maxsize=1000, ttl_seconds=300, shared_cache=True)
async def _fetch_user_tier(user_id: str) -> SubscriptionTier:
    """Fetch one authoritative tier; failed lookups are not cached."""
    try:
        tier = await get_database_manager_async_client().get_user_subscription_tier(
            user_id
        )
    except UnhealthyServiceError:
        # This subclasses ValueError, so it must precede missing-user translation.
        raise
    except ValueError as exc:
        raise SubscriptionTierUserNotFoundError(user_id) from exc
    return SubscriptionTier(tier)


async def get_user_subscription_tier(user_id: str) -> SubscriptionTier:
    """Resolve the authoritative tier through the shared cache."""
    tier = await _fetch_user_tier(user_id)
    # Older pods cached their module-local enum under this same Redis key.
    return SubscriptionTier(tier.value)


get_user_subscription_tier.cache_clear = _fetch_user_tier.cache_clear  # type: ignore[attr-defined]
get_user_subscription_tier.cache_delete = _fetch_user_tier.cache_delete  # type: ignore[attr-defined]


def invalidate_user_subscription_tier(user_id: str) -> None:
    """Best-effort eviction after an authoritative tier write."""
    try:
        _fetch_user_tier.cache_delete(user_id)
    except Exception:
        logger.warning(
            "Failed to invalidate subscription tier cache for user %s",
            user_id[:8],
            exc_info=True,
        )


def subscription_tier_rank(tier: SubscriptionTier) -> int:
    return _SUBSCRIPTION_TIER_RANK[SubscriptionTier(tier)]


def subscription_tier_at_least(
    tier: SubscriptionTier, minimum_tier: SubscriptionTier
) -> bool:
    return subscription_tier_rank(tier) >= subscription_tier_rank(minimum_tier)
