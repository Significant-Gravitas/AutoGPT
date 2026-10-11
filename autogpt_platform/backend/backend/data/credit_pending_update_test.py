"""A failed payment on a change Stripe holds until it's paid changed nothing,
so it must not end the plans a lapsed payment otherwise ends. A trial that
converts early waits like this for the customer's bank to confirm."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data import credit


def _failed_invoice(billing_reason: str) -> dict:
    return {
        "id": "in_held",
        "customer": "cus_1",
        "subscription": "sub_trial",
        "amount_due": 32000,
        "billing_reason": billing_reason,
    }


@pytest.fixture
def lapse():
    user = MagicMock(spec=User)
    user.id = "user-1"
    user.subscriptionTier = SubscriptionTier.TRIAL
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
        patch.object(stripe.Invoice, "pay_async", new_callable=AsyncMock),
    ):
        yield MagicMock(holds=holds, deduct=deduct, cancel=cancel, tier=tier)


@pytest.mark.asyncio
async def test_a_failed_payment_on_a_held_change_cancels_nothing(lapse):
    await credit.handle_subscription_payment_failure(
        _failed_invoice("subscription_update")
    )

    lapse.holds.assert_awaited_once_with("sub_trial", "in_held")
    lapse.deduct.assert_not_awaited()
    lapse.cancel.assert_not_awaited()
    lapse.tier.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_renewal_is_never_looked_up_as_a_held_change(lapse):
    """A pending update waits on the invoice its own change made, so a renewal
    goes straight to the balance without a Stripe lookup."""
    await credit.handle_subscription_payment_failure(
        _failed_invoice("subscription_cycle")
    )

    lapse.holds.assert_not_awaited()
    lapse.deduct.assert_awaited_once()


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
