from unittest.mock import AsyncMock, patch

import pytest
import stripe

from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_invoice import (
    TrialPaymentDue,
    get_trial_payment_due,
)

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

PAYMENT_URL = "https://invoice.stripe.com/i/acct_1/test_inv_1"


@pytest.fixture
def past_due(trial: TrialState) -> TrialState:
    return trial.model_copy(update={"status": "past_due", "subscription_id": "sub_1"})


@pytest.fixture
def subscription(past_due: TrialState) -> dict:
    return {
        "id": "sub_1",
        "customer": past_due.customer_id,
        "status": "past_due",
        "metadata": {"user_id": past_due.user_id, "trial_enrollment_id": past_due.id},
        "latest_invoice": {
            "id": "in_1",
            "created": 1_760_270_400,
            "customer": past_due.customer_id,
            "status": "open",
            "amount_due": 5413,
            "currency": "usd",
            "hosted_invoice_url": PAYMENT_URL,
        },
    }


@pytest.fixture
def retrieve(subscription: dict):
    with patch.object(
        stripe.Subscription, "retrieve_async", AsyncMock(return_value=subscription)
    ) as lookup:
        yield lookup


@pytest.mark.asyncio
async def test_past_due_trial_gets_its_open_invoice(past_due, retrieve):
    due = await get_trial_payment_due(past_due)
    assert due == TrialPaymentDue(
        amount_due=5413, currency="usd", payment_url=PAYMENT_URL
    )
    retrieve.assert_awaited_once_with("sub_1", expand=["latest_invoice"])


@pytest.mark.asyncio
async def test_unpaid_trial_gets_its_open_invoice(past_due, retrieve):
    due = await get_trial_payment_due(past_due.model_copy(update={"status": "unpaid"}))
    assert due is not None and due.payment_url == PAYMENT_URL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [
        {"status": "trialing"},
        {"status": "active"},
        {"status": "canceled"},
        {"subscription_id": None},
    ],
)
async def test_no_lookup_without_a_failed_conversion(past_due, retrieve, update):
    assert await get_trial_payment_due(past_due.model_copy(update=update)) is None
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_lookup_after_conversion(past_due, retrieve):
    converted = past_due.model_copy(
        update={"converted_at": past_due.offer.new_users_from}
    )
    assert await get_trial_payment_due(converted) is None
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoice_update",
    [
        {"status": "paid"},
        {"status": "void"},
        {"status": "draft"},
        {"status": "uncollectible"},
        {"customer": "cus_other"},
        {"amount_due": 0},
        {"hosted_invoice_url": None},
    ],
)
async def test_only_an_open_invoice_owned_by_the_trial_customer(
    past_due, subscription, retrieve, invoice_update
):
    subscription["latest_invoice"].update(invoice_update)
    assert await get_trial_payment_due(past_due) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subscription_update",
    [
        {"customer": "cus_other"},
        {"metadata": {"trial_enrollment_id": "trial-other"}},
        {"latest_invoice": None},
    ],
)
async def test_only_the_trial_enrollment_subscription(
    past_due, subscription, retrieve, subscription_update
):
    subscription.update(subscription_update)
    assert await get_trial_payment_due(past_due) is None


@pytest.mark.asyncio
async def test_stripe_error_leaves_the_status_readable(past_due):
    with patch.object(
        stripe.Subscription,
        "retrieve_async",
        AsyncMock(side_effect=stripe.APIConnectionError("timeout")),
    ):
        assert await get_trial_payment_due(past_due) is None
