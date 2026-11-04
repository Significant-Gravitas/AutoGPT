"""Otto's limits under the entitled tier, and the per-call allowance check for
work billed to the platform's own key."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
)
from backend.data import entitlement_limits
from backend.data.entitlement_limits import platform_spend_refusal

_MOD = "backend.data.entitlement_limits"


def _team_org(decides: bool):
    return patch(f"{_MOD}.org_plan_decides", AsyncMock(return_value=decides))


def _allowance(limits: tuple[int, int, SubscriptionTier], check: AsyncMock):
    return (
        patch(f"{_MOD}.get_global_rate_limits", AsyncMock(return_value=limits)),
        patch(f"{_MOD}.check_rate_limit", check),
    )


@pytest.mark.asyncio
async def test_a_personal_org_keeps_the_users_own_limits():
    own = AsyncMock(return_value=(3, 6, SubscriptionTier.PRO))
    with _team_org(False), patch.object(
        entitlement_limits.rate_limit, "get_global_rate_limits", own
    ):
        limits = await entitlement_limits.get_global_rate_limits("u-1", 1, 2, "org-p")
    assert limits == (3, 6, SubscriptionTier.PRO)
    own.assert_awaited_once_with("u-1", 1, 2)


@pytest.mark.asyncio
async def test_a_team_org_gives_each_member_the_plans_tier_limits():
    by_tier = AsyncMock(return_value=(60, 120, SubscriptionTier.BUSINESS))
    with (
        _team_org(True),
        patch(
            f"{_MOD}.get_user_tier", AsyncMock(return_value=SubscriptionTier.BUSINESS)
        ),
        patch.object(entitlement_limits.rate_limit, "rate_limits_for_tier", by_tier),
    ):
        limits = await entitlement_limits.get_global_rate_limits("u-1", 1, 2, "org-t")
    assert limits == (60, 120, SubscriptionTier.BUSINESS)
    by_tier.assert_awaited_once_with("u-1", SubscriptionTier.BUSINESS, 1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("team", [False, True])
async def test_the_window_check_skips_the_users_own_tier_in_a_team_org(team: bool):
    check = AsyncMock()
    with _team_org(team), patch.object(
        entitlement_limits.rate_limit, "check_rate_limit", check
    ):
        await entitlement_limits.check_rate_limit(
            "u-1", 3, 6, org_id="org", expected_tier=SubscriptionTier.PRO
        )
    assert check.await_args.kwargs["org_plan"] is team


@pytest.mark.asyncio
async def test_storage_follows_the_plans_tier_in_a_team_org():
    by_tier = AsyncMock(return_value=15 * 1024**3)
    with (
        _team_org(True),
        patch(
            f"{_MOD}.get_user_tier", AsyncMock(return_value=SubscriptionTier.BUSINESS)
        ),
        patch.object(
            entitlement_limits.rate_limit, "storage_limit_bytes_for_tier", by_tier
        ),
    ):
        limit = await entitlement_limits.get_workspace_storage_limit_bytes("u-1", "o")
    assert limit == 15 * 1024**3
    by_tier.assert_awaited_once_with(SubscriptionTier.BUSINESS)


@pytest.mark.asyncio
async def test_storage_without_an_org_is_the_users_own():
    own = AsyncMock(return_value=250 * 1024**2)
    with patch.object(
        entitlement_limits.rate_limit, "get_workspace_storage_limit_bytes", own
    ):
        assert await entitlement_limits.get_workspace_storage_limit_bytes("u-1") == (
            250 * 1024**2
        )
    own.assert_awaited_once_with("u-1")


@pytest.mark.asyncio
async def test_a_user_under_their_limits_may_spend():
    check = AsyncMock()
    limits_patch, check_patch = _allowance(
        (3_000_000, 6_000_000, SubscriptionTier.PRO), check
    )
    with limits_patch, check_patch:
        assert await platform_spend_refusal("u-1", "org-1") is None
    check.assert_awaited_once_with(
        user_id="u-1",
        daily_cost_limit=3_000_000,
        weekly_cost_limit=6_000_000,
        expected_tier=SubscriptionTier.PRO,
        org_id="org-1",
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
