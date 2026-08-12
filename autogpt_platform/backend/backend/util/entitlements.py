from enum import Enum
from types import MappingProxyType
from typing import Mapping

from prisma.enums import SubscriptionTier
from pydantic import BaseModel, ConfigDict

from backend.util.settings import BehaveAs, Settings
from backend.util.subscription_tier_order import subscription_tier_at_least
from backend.util.subscription_tiers import (
    SubscriptionTierUserNotFoundError,
    get_authoritative_subscription_tier,
)


class Entitlement(str, Enum):
    CODEX_SUBSCRIPTION_TRANSPORT = "codex_subscription_transport"


class EntitlementPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    minimum_tier: SubscriptionTier
    allow_local: bool = False


ENTITLEMENT_POLICIES: Mapping[Entitlement, EntitlementPolicy] = MappingProxyType(
    {
        Entitlement.CODEX_SUBSCRIPTION_TRANSPORT: EntitlementPolicy(
            minimum_tier=SubscriptionTier.MAX,
            allow_local=True,
        ),
    }
)

settings = Settings()


class EntitlementRequiredError(Exception):
    def __init__(self, entitlement: Entitlement, minimum_tier: SubscriptionTier):
        self.entitlement = entitlement
        self.minimum_tier = minimum_tier
        super().__init__(
            f"{entitlement.value} requires a {minimum_tier.value} plan or higher"
        )


async def _get_user_subscription_tier(user_id: str) -> SubscriptionTier:
    try:
        return await get_authoritative_subscription_tier(user_id)
    except SubscriptionTierUserNotFoundError:
        return SubscriptionTier.NO_TIER


async def has_entitlement(user_id: str, entitlement: Entitlement) -> bool:
    """Check one centralized entitlement policy against the user's DB tier."""
    policy = ENTITLEMENT_POLICIES[entitlement]
    if policy.allow_local and settings.config.behave_as == BehaveAs.LOCAL:
        return True

    tier = await _get_user_subscription_tier(user_id)
    return subscription_tier_at_least(tier, policy.minimum_tier)


async def require_entitlement(user_id: str, entitlement: Entitlement) -> None:
    """Raise when the centralized entitlement policy is not satisfied."""
    policy = ENTITLEMENT_POLICIES[entitlement]
    if not await has_entitlement(user_id, entitlement):
        raise EntitlementRequiredError(entitlement, policy.minimum_tier)
