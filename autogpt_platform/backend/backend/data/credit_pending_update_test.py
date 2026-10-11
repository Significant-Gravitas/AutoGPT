"""A failed payment on a change Stripe holds until it's paid changed nothing,
so it must not end the plans a lapsed payment otherwise ends. A trial that
converts early waits like this for the customer's bank to confirm."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data import credit


@pytest.mark.asyncio
async def test_a_failed_payment_on_a_held_change_cancels_nothing():
    user = MagicMock(spec=User)
    user.id = "user-1"
    user.subscriptionTier = SubscriptionTier.TRIAL
    invoice = {
        "id": "in_held",
        "customer": "cus_1",
        "subscription": "sub_trial",
        "amount_due": 32000,
        "billing_reason": "subscription_update",
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=user)),
        ),
        patch.object(
            credit, "_holds_pending_update", AsyncMock(return_value=True)
        ) as holds,
        patch.object(
            credit.UserCredit, "_add_transaction", new_callable=AsyncMock
        ) as deduct,
        patch.object(
            credit, "_cancel_customer_subscriptions", new_callable=AsyncMock
        ) as cancel,
        patch.object(credit, "set_subscription_tier", new_callable=AsyncMock) as tier,
    ):
        await credit.handle_subscription_payment_failure(invoice)

    holds.assert_awaited_once_with("sub_trial", "in_held")
    deduct.assert_not_awaited()
    cancel.assert_not_awaited()
    tier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subscription,holds",
    [
        ({"pending_update": {"expires_at": 1}, "latest_invoice": "in_held"}, True),
        # A later invoice on the same subscription is not the held one.
        ({"pending_update": {"expires_at": 1}, "latest_invoice": "in_next"}, False),
        ({"pending_update": None, "latest_invoice": "in_held"}, False),
    ],
)
async def test_only_the_invoice_a_pending_update_waits_on_is_held(subscription, holds):
    retrieve = AsyncMock(
        return_value=stripe.Subscription.construct_from(
            {"id": "sub_trial", **subscription}, "test-key"
        )
    )
    with patch.object(stripe.Subscription, "retrieve_async", retrieve):
        assert await credit._holds_pending_update("sub_trial", "in_held") is holds

    retrieve.assert_awaited_once_with("sub_trial")
