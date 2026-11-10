"""The per-call allowance check for work billed to the platform's own key."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.limit_tracking import LimitReading
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
        assert await platform_spend_refusal("u-1", surface="block") is None
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
        refusal = await platform_spend_refusal("u-1", surface="block")
    assert refusal is not None and expected in refusal


@pytest.mark.asyncio
async def test_a_plan_with_no_allowance_refuses_without_reading_the_counters():
    check = AsyncMock()
    limits_patch, check_patch = _allowance((0, 0, SubscriptionTier.NO_TIER), check)
    with limits_patch, check_patch:
        assert await platform_spend_refusal("u-1", surface="block") is not None
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_user_means_nothing_to_meter():
    limits = AsyncMock()
    with patch(f"{_MOD}.get_global_rate_limits", limits):
        assert await platform_spend_refusal(None, surface="block") is None
    limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_spent_allowance_tells_posthog_where_it_refused(chat_blocked_events):
    resets_at = datetime.now(UTC) + timedelta(hours=2)
    error = RateLimitExceeded(
        "daily", resets_at, reading=LimitReading(tier="PRO", used=3.2, cap=3.0)
    )
    limits_patch, check_patch = _allowance(
        (3_000_000, 6_000_000, SubscriptionTier.PRO), AsyncMock(side_effect=error)
    )
    with limits_patch, check_patch:
        await platform_spend_refusal("u-1", surface="block")
    assert chat_blocked_events == [
        {
            "distinct_id": "u-1",
            "reason": "usage_limit",
            "limit": "daily",
            "outcome": "refused",
            "surface": "block",
            "tier": "PRO",
            "used": 3.2,
            "cap": 3.0,
            "resets_at": resets_at.isoformat(),
        }
    ]


@pytest.mark.asyncio
async def test_a_plan_with_no_allowance_is_a_weekly_window_that_allows_nothing(
    chat_blocked_events,
):
    limits_patch, check_patch = _allowance(
        (0, 0, SubscriptionTier.NO_TIER), AsyncMock()
    )
    with limits_patch, check_patch:
        await platform_spend_refusal("u-1", surface="chat")
    (event,) = chat_blocked_events
    assert (event["limit"], event["cap"], event["resets_at"]) == ("weekly", 0.0, None)
    assert event["tier"] == SubscriptionTier.NO_TIER.value


@pytest.mark.asyncio
async def test_an_unreadable_allowance_is_not_a_limit(chat_blocked_events):
    limits_patch, check_patch = _allowance(
        (3_000_000, 6_000_000, SubscriptionTier.PRO),
        AsyncMock(side_effect=RateLimitUnavailable("redis down")),
    )
    with limits_patch, check_patch:
        assert await platform_spend_refusal("u-1", surface="block") is not None
    assert chat_blocked_events == []
