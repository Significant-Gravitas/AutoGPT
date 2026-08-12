from prisma.enums import SubscriptionTier

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


def subscription_tier_rank(tier: SubscriptionTier) -> int:
    return _SUBSCRIPTION_TIER_RANK[tier]


def subscription_tier_at_least(
    tier: SubscriptionTier,
    minimum_tier: SubscriptionTier,
) -> bool:
    return subscription_tier_rank(tier) >= subscription_tier_rank(minimum_tier)
