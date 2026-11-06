"""Which open bill the paywall offers a customer whose renewal failed."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data.subscription_payment_due import get_subscription_payment_due

CUSTOMER = "cus_1"
PAY_URL = "https://invoice.stripe.com/i/acct_1/test_bill"


def _subscription(sub_id: str = "sub_1", **invoice) -> dict:
    return {
        "id": sub_id,
        "customer": CUSTOMER,
        "status": "past_due",
        "created": 100,
        "current_period_start": 1772539200,
        "pending_update": None,
        "items": {"data": [{"price": {"id": "price_pro"}}]},
        "latest_invoice": {
            "id": f"in_{sub_id}",
            "customer": CUSTOMER,
            "status": "open",
            "amount_due": 5000,
            "currency": "usd",
            "hosted_invoice_url": PAY_URL,
            **invoice,
        },
    }


def _page(*subscriptions: dict) -> MagicMock:
    page = MagicMock()
    page.data = list(subscriptions)
    page.has_more = False
    return page


async def _due(*past_due: dict, replaced: bool = False):
    async def list_subscriptions(customer, status, limit, expand):
        assert customer == CUSTOMER and expand == ["data.latest_invoice"]
        return _page(*past_due) if status == "past_due" else _page()

    module = "backend.data.subscription_payment_due"
    with (
        patch("stripe.Subscription.list_async", side_effect=list_subscriptions),
        patch(
            f"{module}.replaced_by_newer_plan",
            new_callable=AsyncMock,
            return_value=replaced,
        ),
        patch(
            f"{module}.build_price_to_tier_map",
            new_callable=AsyncMock,
            return_value={"price_pro": SubscriptionTier.PRO},
        ),
    ):
        return await get_subscription_payment_due(CUSTOMER, fresh=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["open", "uncollectible"])
async def test_a_failed_renewals_payable_bill_is_offered(status):
    """Stripe marks the bill uncollectible once its retries run out; its
    payment link still works, so it is still the way back."""
    due = await _due(_subscription(status=status))

    assert due is not None
    assert due.model_dump() == {
        "amount_due": 5000,
        "currency": "usd",
        "payment_url": PAY_URL,
        "plan": "PRO",
        "due_since": 1772539200,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoice",
    [
        {"status": "paid"},
        {"status": "void"},
        {"amount_due": 0},
        {"hosted_invoice_url": None},
        {"customer": "cus_other"},
    ],
)
async def test_a_bill_that_cannot_be_paid_here_is_not_offered(invoice):
    assert await _due(_subscription(**invoice)) is None


@pytest.mark.asyncio
async def test_a_plan_replaced_by_a_newer_one_is_not_offered():
    assert await _due(_subscription(), replaced=True) is None


@pytest.mark.asyncio
async def test_a_held_upgrade_is_not_a_bill_to_restore_the_plan():
    held = _subscription()
    held["pending_update"] = {"expires_at": 1}
    assert await _due(held) is None


@pytest.mark.asyncio
async def test_no_customer_reads_nothing_from_stripe():
    with patch("stripe.Subscription.list_async") as listing:
        assert await get_subscription_payment_due(None) is None
    listing.assert_not_called()
