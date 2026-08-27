import logging
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from prisma.enums import SubscriptionTier
from pydantic import BaseModel, ConfigDict

from backend.util.clients import get_database_manager_async_client
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)


class Entitlement(str, Enum):
    CODEX_SUBSCRIPTION_TRANSPORT = "codex_subscription_transport"
    GITHUB_COPILOT_SUBSCRIPTION_TRANSPORT = "github_copilot_subscription_transport"
    GROK_SUBSCRIPTION_TRANSPORT = "grok_subscription_transport"
    # The Advanced model tier. Gated on hosted so it is a reason to upgrade
    # rather than a reason not to; granted outright on self-host, where
    # there is no plan to sell and the operator pays for their own tokens.
    ADVANCED_MODEL_TIER = "advanced_model_tier"


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
        Entitlement.GITHUB_COPILOT_SUBSCRIPTION_TRANSPORT: EntitlementPolicy(
            minimum_tier=SubscriptionTier.MAX,
            allow_local=True,
        ),
        Entitlement.GROK_SUBSCRIPTION_TRANSPORT: EntitlementPolicy(
            minimum_tier=SubscriptionTier.MAX,
            allow_local=True,
        ),
        Entitlement.ADVANCED_MODEL_TIER: EntitlementPolicy(
            minimum_tier=SubscriptionTier.MAX,
            allow_local=True,
        ),
    }
)

_TIER_ORDER = (
    SubscriptionTier.NO_TIER,
    SubscriptionTier.BASIC,
    SubscriptionTier.PRO,
    SubscriptionTier.MAX,
    SubscriptionTier.BUSINESS,
    SubscriptionTier.ENTERPRISE,
)
_TIER_RANK = {tier: rank for rank, tier in enumerate(_TIER_ORDER)}

settings = Settings()


class EntitlementRequiredError(Exception):
    def __init__(self, entitlement: Entitlement, minimum_tier: SubscriptionTier):
        self.entitlement = entitlement
        self.minimum_tier = minimum_tier
        super().__init__(
            f"{entitlement.value} requires a {minimum_tier.value} plan or higher"
        )


async def _get_user_subscription_tier(user_id: str) -> SubscriptionTier:
    """Resolve a tier through DatabaseManager without caching this result.

    A missing user has no entitlement. Other database and transport failures
    propagate so callers can retry instead of treating an outage as a denial.
    """
    try:
        tier = await get_database_manager_async_client().get_user_subscription_tier(
            user_id
        )
    except ValueError:
        return SubscriptionTier.NO_TIER
    return SubscriptionTier(tier)


async def has_entitlement(user_id: str, entitlement: Entitlement) -> bool:
    """Check one centralized entitlement policy against the user's DB tier."""
    policy = ENTITLEMENT_POLICIES[entitlement]
    if policy.allow_local and settings.config.behave_as == BehaveAs.LOCAL:
        return True

    tier = await _get_user_subscription_tier(user_id)
    return _TIER_RANK[tier] >= _TIER_RANK[policy.minimum_tier]


async def has_entitlement_for_discovery(user_id: str, entitlement: Entitlement) -> bool:
    """Entitlement check for deciding whether to *show* something.

    Fails closed and never raises. A lookup that errors hides the offer
    rather than taking the surface down with it -- listing connections is not
    a good enough reason to fail someone's page, and an offer wrongly hidden
    for one render costs less than one wrongly shown, which would produce a
    connection that refuses on first use.

    Enforcement asks a different question and lives elsewhere: use
    ``require_entitlement`` where the answer decides whether to spend.
    """
    try:
        return await has_entitlement(user_id, entitlement)
    except Exception:
        logger.warning(
            "Could not resolve entitlement %s for user %s; hiding it",
            entitlement,
            user_id,
            exc_info=True,
        )
        return False


async def require_entitlement(user_id: str, entitlement: Entitlement) -> None:
    """Raise when the centralized entitlement policy is not satisfied."""
    policy = ENTITLEMENT_POLICIES[entitlement]
    if not await has_entitlement(user_id, entitlement):
        raise EntitlementRequiredError(entitlement, policy.minimum_tier)
