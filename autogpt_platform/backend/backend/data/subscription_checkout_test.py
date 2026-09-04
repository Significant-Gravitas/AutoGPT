from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit
from backend.data.credit import _expire_open_subscription_sessions


@pytest.mark.asyncio
async def test_expiration_list_failure_stops_new_checkout():
    with patch(
        "backend.data.credit.stripe.checkout.Session.list_async",
        AsyncMock(side_effect=stripe.APIConnectionError("unavailable")),
    ):
        with pytest.raises(stripe.APIConnectionError):
            await _expire_open_subscription_sessions("cus_test")


@pytest.mark.asyncio
async def test_in_place_change_cannot_bypass_trial_with_stale_local_tier():
    subscription = stripe.Subscription.construct_from(
        {
            "id": "sub_trial",
            "status": "trialing",
            "schedule": None,
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_trial"}]},
            "metadata": {"trial_enrollment_id": "trial-1"},
        },
        "test-key",
    )
    user = MagicMock(
        stripe_customer_id="cus_test", subscription_tier=SubscriptionTier.NO_TIER
    )
    with (
        patch.object(
            credit, "get_subscription_price_id", AsyncMock(return_value="price_max")
        ),
        patch.object(credit, "get_user_by_id", AsyncMock(return_value=user)),
        patch.object(
            credit, "_get_active_subscription", AsyncMock(return_value=subscription)
        ),
        patch.object(credit.stripe.Subscription, "modify_async", AsyncMock()) as modify,
        patch.object(credit, "set_subscription_tier", AsyncMock()) as promote,
        patch.object(credit, "_track_billing_event"),
    ):
        with pytest.raises(ValueError, match="trial"):
            await credit.modify_stripe_subscription_for_tier(
                "user-1", SubscriptionTier.MAX
            )
    modify.assert_not_awaited()
    promote.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_completed_during_expiration_stops_new_checkout():
    sessions = stripe.ListObject.construct_from(
        {"data": [{"id": "cs_old", "mode": "subscription"}], "has_more": False},
        "test-key",
    )
    with (
        patch(
            "backend.data.credit.stripe.checkout.Session.list_async",
            AsyncMock(return_value=sessions),
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.expire_async",
            AsyncMock(side_effect=stripe.InvalidRequestError("already complete", "id")),
        ),
    ):
        with pytest.raises(stripe.InvalidRequestError):
            await _expire_open_subscription_sessions("cus_test")
