from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.rate_limit import SubscriptionTier, get_global_rate_limits


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "multiplier,config_daily,config_weekly,expected_daily,expected_weekly",
    [
        (0.5, -1, -1, -1, -1),
        (0.5, -1, 200, -1, 100),
        (0.5, 100, -1, 50, -1),
        (2.0, -1, -1, -1, -1),
        (0.5, 0, 200, 0, 100),
        (0.5, 100, 200, 50, 100),
    ],
)
async def test_tier_scaling_preserves_unlimited_windows(
    multiplier: float,
    config_daily: int,
    config_weekly: int,
    expected_daily: int,
    expected_weekly: int,
):
    with (
        patch(
            "backend.copilot.rate_limit._fetch_cost_limits_flag",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.rate_limit.get_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        ),
        patch(
            "backend.copilot.rate_limit.get_tier_multipliers",
            new=AsyncMock(
                return_value={
                    SubscriptionTier.NO_TIER.value: 0.0,
                    SubscriptionTier.BASIC.value: multiplier,
                }
            ),
        ),
        patch(
            "backend.copilot.rate_limit.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ),
    ):
        daily, weekly, tier = await get_global_rate_limits(
            "unlimited-user", config_daily, config_weekly
        )

    assert (daily, weekly, tier) == (
        expected_daily,
        expected_weekly,
        SubscriptionTier.NO_TIER,
    )
