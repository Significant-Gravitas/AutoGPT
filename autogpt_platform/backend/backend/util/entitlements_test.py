from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.util import entitlements
from backend.util.entitlements import Entitlement, EntitlementRequiredError
from backend.util.settings import BehaveAs


def _database_client(*tiers: SubscriptionTier | Exception):
    client = MagicMock()
    client.get_user_subscription_tier = AsyncMock(side_effect=tiers)
    return client


@pytest.mark.asyncio
async def test_local_entitlement_bypasses_database_manager():
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.LOCAL),
        patch.object(entitlements, "get_database_manager_async_client") as get_db,
    ):
        assert await entitlements.has_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )
        await entitlements.require_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )

    get_db.assert_not_called()


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
    client = _database_client(tier)
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_database_manager_async_client",
            return_value=client,
        ),
    ):
        result = await entitlements.has_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )

    assert result is allowed
    client.get_user_subscription_tier.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_repeated_checks_repeat_database_manager_lookup():
    client = _database_client(SubscriptionTier.MAX, SubscriptionTier.PRO)
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_database_manager_async_client",
            return_value=client,
        ),
    ):
        first = await entitlements.has_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )
        second = await entitlements.has_entitlement(
            "user-1",
            Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
        )

    assert first is True
    assert second is False
    assert client.get_user_subscription_tier.await_count == 2


@pytest.mark.asyncio
async def test_require_entitlement_raises_below_minimum_tier():
    client = _database_client(SubscriptionTier.PRO)
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_database_manager_async_client",
            return_value=client,
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
    client = _database_client(RuntimeError("database unavailable"))
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_database_manager_async_client",
            return_value=client,
        ),
    ):
        with pytest.raises(RuntimeError, match="database unavailable"):
            await entitlements.has_entitlement(
                "user-1",
                Entitlement.CODEX_SUBSCRIPTION_TRANSPORT,
            )


@pytest.mark.asyncio
async def test_missing_user_is_treated_as_no_tier():
    client = _database_client(ValueError("User not found"))
    with (
        patch.object(entitlements.settings.config, "behave_as", BehaveAs.CLOUD),
        patch.object(
            entitlements,
            "get_database_manager_async_client",
            return_value=client,
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
