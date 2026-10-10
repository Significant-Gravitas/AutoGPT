"""Stripe subscription sync and pending-change lookups for a trial whose
cancellation is scheduled for the end of the trial."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit_subscription_test
from backend.data.credit import (
    get_pending_subscription_change,
    sync_subscription_from_stripe,
)
from backend.data.credit_subscription_test import _clear_cache, _make_user

_clear_active_subscription_cache = (
    credit_subscription_test._clear_active_subscription_cache
)


@pytest.mark.asyncio
async def test_new_plan_ends_cancel_pending_trial_without_invoice():
    """A plan bought while the trial is cancel-pending ends the trial at once,
    with no invoice or proration, once the new subscription is active."""
    metadata = {"user_id": "user-1"}
    max_sub = {
        "id": "sub_max",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_max_monthly"}}]},
        "metadata": metadata,
    }
    trial_sub = {
        "id": "sub_trial",
        "customer": "cus_123",
        "status": "trialing",
        "cancel_at_period_end": True,
        "metadata": {**metadata, "trial_enrollment_id": "trial-1"},
    }
    ended_trial = {**trial_sub, "status": "canceled"}
    events: list[str] = []

    async def list_subscriptions(*, customer: str, status: str, limit: int):
        return MagicMock(
            data=[max_sub] if status == "active" else [trial_sub], has_more=False
        )

    async def cancel(sub_id: str, **params):
        events.append(f"cancel:{sub_id}")
        return ended_trial

    async def set_tier(user_id: str, tier: SubscriptionTier, **kwargs):
        events.append(f"tier:{tier.value}")

    async def price_id(tier: SubscriptionTier, billing_cycle: str = "monthly"):
        return {SubscriptionTier.MAX: "price_max_monthly"}.get(tier)

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(
                find_first=AsyncMock(
                    return_value=_make_user(tier=SubscriptionTier.TRIAL)
                )
            ),
        ),
        patch("backend.data.credit.get_subscription_price_id", side_effect=price_id),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            side_effect=list_subscriptions,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            new_callable=AsyncMock,
            side_effect=cancel,
        ) as mock_cancel,
        patch(
            "backend.data.credit.reconcile_trial_subscription",
            new_callable=AsyncMock,
            return_value=(ended_trial, None),
        ) as reconcile,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
            side_effect=set_tier,
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(max_sub)

    mock_cancel.assert_awaited_once_with("sub_trial", invoice_now=False, prorate=False)
    reconcile.assert_awaited_once_with("user-1", "sub_trial")
    mock_set.assert_awaited_once_with(
        "user-1", SubscriptionTier.MAX, track_lifecycle=False
    )
    assert events == ["cancel:sub_trial", "tier:MAX"]


async def _pending_change_for(subscription: dict):
    _clear_cache(get_pending_subscription_change)
    mock_list = MagicMock()
    mock_list.data = [stripe.Subscription.construct_from(subscription, "k")]
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=MagicMock(stripe_customer_id="cus_abc"),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_pro",
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.retrieve_async",
            new_callable=AsyncMock,
        ) as retrieve_schedule,
    ):
        result = await get_pending_subscription_change("user-1")
    retrieve_schedule.assert_not_awaited()
    return result


@pytest.mark.asyncio
async def test_get_pending_subscription_change_ignores_cancel_pending_trial():
    """A trial scheduled to end has no paid plan to downgrade from."""
    result = await _pending_change_for(
        {
            "id": "sub_trial",
            "status": "trialing",
            "current_period_end": int(time.time()) + 5 * 24 * 3600,
            "cancel_at_period_end": True,
            "schedule": None,
            "metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-1"},
        }
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata", [{}, {"trial_enrollment_id": "trial-1", "user_id": "user-1"}]
)
async def test_get_pending_subscription_change_reports_paid_cancel(metadata):
    """A paid subscription, including a converted trial, still reports its
    scheduled cancellation."""
    period_end = int(time.time()) + 10 * 24 * 3600
    result = await _pending_change_for(
        {
            "id": "sub_pro",
            "status": "active",
            "current_period_end": period_end,
            "cancel_at_period_end": True,
            "schedule": None,
            "metadata": metadata,
        }
    )

    assert result is not None
    pending_tier, effective_at, pending_cycle = result
    assert pending_tier == SubscriptionTier.NO_TIER
    assert int(effective_at.timestamp()) == period_end
    assert pending_cycle is None
