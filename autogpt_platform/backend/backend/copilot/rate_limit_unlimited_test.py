from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
    check_rate_limit,
    get_global_rate_limits,
)
from backend.data.subscription_trial import TrialState


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


@pytest.mark.asyncio
@pytest.mark.parametrize("daily,skip_daily", [(-1, False), (0, True), (100, True)])
async def test_uncapped_windows_do_not_require_redis(daily: int, skip_daily: bool):
    with (
        patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        ),
        patch(
            "backend.copilot.rate_limit.get_redis_async",
            new=AsyncMock(side_effect=ConnectionError("Redis unavailable")),
        ) as redis,
    ):
        await check_rate_limit("unlimited-user", daily, -1, skip_daily=skip_daily)

    redis.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("daily,weekly,skip_daily", [(0, -1, False), (-1, 0, True)])
async def test_active_window_still_fails_closed_without_redis(
    daily: int, weekly: int, skip_daily: bool
):
    with (
        patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.NO_TIER),
        ),
        patch(
            "backend.copilot.rate_limit.get_redis_async",
            new=AsyncMock(side_effect=ConnectionError("Redis unavailable")),
        ),
        pytest.raises(RateLimitUnavailable),
    ):
        await check_rate_limit("limited-user", daily, weekly, skip_daily=skip_daily)


@pytest.mark.asyncio
@pytest.mark.parametrize("active,cost", [(False, 0), (True, 100)])
async def test_uncapped_windows_preserve_trial_enforcement(active: bool, cost: int):
    trial = MagicMock(spec=TrialState)
    trial.active = active
    trial.cost_microdollars = cost
    trial.offer = MagicMock(total_cost_limit=100)
    trial.ends_at = None
    store = MagicMock()
    store.get_subscription_trial = AsyncMock(return_value=trial)
    with (
        patch(
            "backend.copilot.rate_limit._fetch_user_tier",
            new=AsyncMock(return_value=SubscriptionTier.TRIAL),
        ),
        patch("backend.copilot.rate_limit.credit_db", return_value=store),
        patch("backend.copilot.rate_limit.get_redis_async", new=AsyncMock()) as redis,
        pytest.raises(RateLimitExceeded) as exc,
    ):
        await check_rate_limit("trial-user", -1, -1)

    assert exc.value.window == "trial"
    redis.assert_not_awaited()
