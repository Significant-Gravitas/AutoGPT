from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data import user as user_module


@pytest.mark.asyncio
async def test_get_user_subscription_tier_is_uncached():
    with patch.object(user_module, "prisma") as mock_prisma:
        mock_prisma.user.find_unique = AsyncMock(
            side_effect=[
                MagicMock(subscriptionTier=SubscriptionTier.PRO),
                MagicMock(subscriptionTier=SubscriptionTier.MAX),
            ]
        )

        first = await user_module.get_user_subscription_tier("user-1")
        second = await user_module.get_user_subscription_tier("user-1")

    assert first == SubscriptionTier.PRO
    assert second == SubscriptionTier.MAX
    assert mock_prisma.user.find_unique.await_count == 2


@pytest.mark.asyncio
async def test_get_user_subscription_tier_defaults_null_to_no_tier():
    with patch.object(user_module, "prisma") as mock_prisma:
        mock_prisma.user.find_unique = AsyncMock(
            return_value=MagicMock(subscriptionTier=None)
        )

        tier = await user_module.get_user_subscription_tier("user-1")

    assert tier == SubscriptionTier.NO_TIER


@pytest.mark.asyncio
async def test_get_user_subscription_tier_raises_for_missing_user():
    with patch.object(user_module, "prisma") as mock_prisma:
        mock_prisma.user.find_unique = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="User not found"):
            await user_module.get_user_subscription_tier("missing-user")
