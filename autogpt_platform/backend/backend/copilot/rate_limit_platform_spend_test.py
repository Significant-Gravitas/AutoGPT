"""The per-call allowance check for work billed to the platform's own key."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
    platform_spend_refusal,
)

_MOD = "backend.copilot.rate_limit"


def _allowance(limits: tuple[int, int, SubscriptionTier], check: AsyncMock):
    return (
        patch(f"{_MOD}.get_global_rate_limits", AsyncMock(return_value=limits)),
        patch(f"{_MOD}.check_rate_limit", check),
    )


@pytest.mark.asyncio
async def test_a_user_under_their_limits_may_spend():
    check = AsyncMock()
    limits_patch, check_patch = _allowance(
        (3_000_000, 6_000_000, SubscriptionTier.PRO), check
    )
    with limits_patch, check_patch:
        assert await platform_spend_refusal("u-1") is None
    check.assert_awaited_once_with(
        user_id="u-1",
        daily_cost_limit=3_000_000,
        weekly_cost_limit=6_000_000,
        expected_tier=SubscriptionTier.PRO,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, expected",
    [
        (
            RateLimitExceeded("daily", datetime.now(UTC) + timedelta(hours=2)),
            "daily usage limit",
        ),
        (RateLimitUnavailable("redis down"), "can't be checked"),
    ],
    ids=["limit-reached", "limits-unreadable"],
)
async def test_a_spent_or_unreadable_allowance_refuses(error, expected):
    limits_patch, check_patch = _allowance(
        (3_000_000, 6_000_000, SubscriptionTier.PRO), AsyncMock(side_effect=error)
    )
    with limits_patch, check_patch:
        refusal = await platform_spend_refusal("u-1")
    assert refusal is not None and expected in refusal


@pytest.mark.asyncio
async def test_a_plan_with_no_allowance_refuses_without_reading_the_counters():
    check = AsyncMock()
    limits_patch, check_patch = _allowance((0, 0, SubscriptionTier.NO_TIER), check)
    with limits_patch, check_patch:
        assert await platform_spend_refusal("u-1") is not None
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_user_means_nothing_to_meter():
    limits = AsyncMock()
    with patch(f"{_MOD}.get_global_rate_limits", limits):
        assert await platform_spend_refusal(None) is None
    limits.assert_not_awaited()
