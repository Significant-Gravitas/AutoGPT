from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.util import entitlements
from backend.util.entitlements import Entitlement, EntitlementRequiredError
from backend.util.settings import BehaveAs
from backend.util.subscription_tiers import SubscriptionTierUserNotFoundError


@pytest.mark.asyncio
async def test_local_entitlement_bypasses_tier_resolver():
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.LOCAL),
        patch.object(entitlements, "get_authoritative_subscription_tier") as resolve,
    ):
        assert await entitlements.has_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )
        await entitlements.require_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )

    resolve.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tier", "allowed"),
    [
        (SubscriptionTier.NO_TIER, False),
        (SubscriptionTier.BASIC, False),
        (SubscriptionTier.PRO, False),
        (SubscriptionTier.MAX, True),
        (SubscriptionTier.BUSINESS, True),
        (SubscriptionTier.ENTERPRISE, True),
    ],
)
async def test_cloud_entitlement_uses_minimum_tier_order(
    tier: SubscriptionTier,
    allowed: bool,
):
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_authoritative_subscription_tier",
            new=AsyncMock(return_value=tier),
        ),
    ):
        result = await entitlements.has_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )

    assert result is allowed


@pytest.mark.asyncio
async def test_require_entitlement_raises_below_minimum_tier():
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_authoritative_subscription_tier",
            new=AsyncMock(return_value=SubscriptionTier.PRO),
        ),
    ):
        with pytest.raises(EntitlementRequiredError) as exc_info:
            await entitlements.require_entitlement(
                "user-1",
                Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
            )

    assert exc_info.value.entitlement == Entitlement.CODEX_SUBSCRIPTION_TRANSPORT
    assert exc_info.value.minimum_tier == SubscriptionTier.MAX


@pytest.mark.asyncio
async def test_database_manager_error_propagates():
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_authoritative_subscription_tier",
            new=AsyncMock(side_effect=RuntimeError("database unavailable")),
        ),
    ):
        with pytest.raises(RuntimeError, match="database unavailable"):
            await entitlements.has_entitlement(
                "user-1",
                Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
            )


@pytest.mark.asyncio
async def test_missing_user_is_treated_as_no_tier():
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_authoritative_subscription_tier",
            new=AsyncMock(side_effect=SubscriptionTierUserNotFoundError("missing")),
        ),
    ):
        allowed = await entitlements.has_entitlement(
            "missing-user",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )

    assert allowed is False


def test_codex_policy_is_centralized_as_max_plus_with_local_access():
    policy = entitlements.ENTITLEMENT_POLICIES[Entitlement.CODEX_SUBSCRIPTION_TRANSPORT]

    assert policy.minimum_tier == SubscriptionTier.MAX
    assert policy.allow_local is True
