from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data import stripe_reconciliation as sweep


@pytest.fixture(autouse=True)
def activation_rows(mocker):
    rows = MagicMock(find_unique=AsyncMock(return_value=None))
    mocker.patch.object(sweep.PaidUsageActivation, "prisma", return_value=rows)
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [SubscriptionTier.PRO, SubscriptionTier.MAX])
@pytest.mark.parametrize("unchanged", [False, True])
async def test_paid_sweep_reconciles_even_unchanged_tier(target, unchanged):
    initial = target if unchanged else SubscriptionTier.NO_TIER
    before = MagicMock(id="user", stripeCustomerId="cus", subscriptionTier=initial)
    after = MagicMock(subscriptionTier=target)
    raw = {"id": "sub", "customer": "cus"}
    summary = sweep.ReconciliationSummary()
    with (
        patch.object(sweep, "sync_subscription_from_stripe", AsyncMock()) as sync,
        patch.object(sweep, "set_subscription_tier", AsyncMock()) as direct,
        patch.object(
            sweep.User,
            "prisma",
            return_value=MagicMock(find_unique_or_raise=AsyncMock(return_value=after)),
        ),
        patch.object(
            sweep, "log_tier_reconciliation_discrepancy", return_value="upgrade"
        ),
    ):
        await sweep._reconcile_one(before, {"cus": target}, summary, True, {"cus": raw})
    sync.assert_awaited_once_with(raw, track_lifecycle=False)
    direct.assert_not_awaited()
    assert summary.upgrades == (initial == SubscriptionTier.NO_TIER)
    assert summary.unchanged == unchanged


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [SubscriptionTier.PRO, SubscriptionTier.MAX])
async def test_paid_sweep_does_not_grant_access_when_invoice_is_unsettled(target):
    before = MagicMock(
        id="user", stripeCustomerId="cus", subscriptionTier=SubscriptionTier.NO_TIER
    )
    summary = sweep.ReconciliationSummary()
    raw = {"id": "sub", "customer": "cus", "status": "active"}
    with (
        patch.object(sweep, "sync_subscription_from_stripe", AsyncMock()) as sync,
        patch.object(sweep, "set_subscription_tier", AsyncMock()) as direct,
        patch.object(
            sweep.User,
            "prisma",
            return_value=MagicMock(find_unique_or_raise=AsyncMock(return_value=before)),
        ),
    ):
        await sweep._reconcile_one(before, {"cus": target}, summary, True, {"cus": raw})
    sync.assert_awaited_once_with(raw, track_lifecycle=False)
    direct.assert_not_awaited()
    assert summary.unchanged == 1
    assert summary.upgrades == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ConnectionError("Redis unavailable"), ValueError("invalid evidence")]
)
async def test_pro_sweep_failure_never_falls_back_to_direct_grant(failure):
    user = MagicMock(
        id="user", stripeCustomerId="cus", subscriptionTier=SubscriptionTier.NO_TIER
    )
    summary = sweep.ReconciliationSummary()
    with (
        patch.object(
            sweep, "sync_subscription_from_stripe", AsyncMock(side_effect=failure)
        ),
        patch.object(sweep, "set_subscription_tier", AsyncMock()) as direct,
    ):
        await sweep._reconcile_one(
            user,
            {"cus": SubscriptionTier.PRO},
            summary,
            True,
            {"cus": {"id": "sub", "customer": "cus"}},
        )
    direct.assert_not_awaited()
    assert summary.errors == 1
    assert summary.upgrades == 0


@pytest.mark.asyncio
async def test_pro_sweep_missing_subscription_identity_fails_closed():
    user = MagicMock(
        id="user", stripeCustomerId="cus", subscriptionTier=SubscriptionTier.NO_TIER
    )
    summary = sweep.ReconciliationSummary()
    with patch.object(sweep, "set_subscription_tier", AsyncMock()) as direct:
        await sweep._reconcile_one(user, {"cus": SubscriptionTier.PRO}, summary, True)
    direct.assert_not_awaited()
    assert summary.errors == 1


def test_pro_candidate_preserves_subscription_identity_alongside_highest_tier():
    tiers = {}
    subscriptions = {}
    prices = {"pro": SubscriptionTier.PRO, "basic": SubscriptionTier.BASIC}
    pro = {
        "id": "sub_pro",
        "customer": "cus",
        "items": {"data": [{"price": {"id": "pro"}}]},
    }
    basic = {
        "id": "sub_basic",
        "customer": "cus",
        "items": {"data": [{"price": {"id": "basic"}}]},
    }
    sweep._record_subscription(pro, prices, tiers, subscriptions)
    sweep._record_subscription(basic, prices, tiers, subscriptions)
    assert tiers == {"cus": SubscriptionTier.PRO}
    assert subscriptions == {"cus": pro}


@pytest.mark.parametrize("max_first", [True, False])
def test_max_candidate_keeps_its_identity_over_lower_paid_subscription(max_first):
    tiers, subscriptions = {}, {}
    prices = {"pro": SubscriptionTier.PRO, "max": SubscriptionTier.MAX}
    pro, maximum = (
        {
            "id": f"sub_{tier}",
            "customer": "cus",
            "items": {"data": [{"price": {"id": tier}}]},
        }
        for tier in ("pro", "max")
    )
    for subscription in (maximum, pro) if max_first else (pro, maximum):
        sweep._record_subscription(subscription, prices, tiers, subscriptions)
    assert tiers == {"cus": SubscriptionTier.MAX}
    assert subscriptions == {"cus": maximum}


@pytest.mark.asyncio
async def test_completed_generation_does_not_skip_paid_plan_change(activation_rows):
    activation_rows.find_unique.return_value = MagicMock(
        stripeSubscriptionId="sub", readyAt="ready"
    )
    user = MagicMock(id="user", subscriptionTier=SubscriptionTier.PRO)
    assert await sweep._requires_pro_reconciliation(
        user, {"id": "sub"}, SubscriptionTier.MAX
    )
