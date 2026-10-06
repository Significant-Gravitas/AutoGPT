"""Real PostgreSQL confirmation durability and cross-request serialization."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import stripe
from prisma.models import User

from backend.data import subscription_activation_checkout as checkout
from backend.data.subscription_activation_attempt import (
    get_attempt,
    save_confirmation,
    save_quote,
)
from backend.data.subscription_activation_models import (
    ActivationConfirmRequest,
    ActivationTerms,
    ActivationUnavailable,
)
from backend.data.subscription_activation_stripe import (
    BillingInvoice,
    BillingSubscription,
)
from backend.data.subscription_checkout import (
    SubscriptionCheckoutUnavailable,
    subscription_checkout_lock,
)

pytest_plugins = ("backend.data.subscription_trial_integration_fixtures",)
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires explicitly selected disposable trial database",
)


def terms():
    return ActivationTerms(
        price_id="price_test",
        accepted_offer_token="a" * 64,
        amount_due=2000,
        currency="usd",
        billing_interval="month",
        renewal_unit_amount=2000,
        renewal_terms="Renews monthly until canceled.",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )


@pytest.mark.asyncio
async def test_replaced_quote_cannot_be_confirmed_by_a_stale_request(enrollment):
    original = await save_quote(
        enrollment.user_id,
        f"sub_{enrollment.id}",
        enrollment.customer_id,
        terms(),
        "/chat/first",
    )
    replacement = await save_quote(
        enrollment.user_id,
        original.subscription_id,
        enrollment.customer_id,
        terms().model_copy(update={"amount_due": 2500}),
        "/chat/second",
    )
    with pytest.raises(ActivationUnavailable, match="terms changed"):
        await save_confirmation(original)
    current = await get_attempt(enrollment.user_id)
    assert current and current.confirmed_at is None
    assert current.terms == replacement.terms


@pytest.mark.asyncio
async def test_confirmed_quote_is_immutable_and_user_scoped(enrollment):
    attempt = await save_quote(
        enrollment.user_id,
        f"sub_{enrollment.id}",
        enrollment.customer_id,
        terms(),
        "/chat/return-here",
    )
    async with subscription_checkout_lock(enrollment.user_id):
        confirmed = await save_confirmation(attempt)
    assert confirmed.confirmed_at is not None
    assert await get_attempt("another-user", attempt.id) is None
    with pytest.raises(ValueError, match="already in progress"):
        await save_quote(
            enrollment.user_id,
            attempt.subscription_id,
            enrollment.customer_id,
            terms().model_copy(update={"amount_due": 3000}),
            "/different-path",
        )
    persisted = await get_attempt(enrollment.user_id)
    assert persisted is not None
    assert persisted.terms.amount_due == 2000
    assert persisted.return_to == "/chat/return-here"


@pytest.mark.asyncio
async def test_concurrent_confirmation_and_lost_stripe_response_do_not_charge_twice(
    enrollment, monkeypatch
):
    await User.prisma().update(
        where={"id": enrollment.user_id},
        data={"stripeCustomerId": enrollment.customer_id},
    )
    attempt = await save_quote(
        enrollment.user_id,
        f"sub_{enrollment.id}",
        enrollment.customer_id,
        terms(),
        "/chat/return-here",
    )
    sub = BillingSubscription.model_validate(
        {
            "id": attempt.subscription_id,
            "customer": enrollment.customer_id,
            "status": "trialing",
            "trial_end": int(datetime.now(UTC).timestamp()) + 86400,
            "metadata": {"user_id": enrollment.user_id},
            "items": {
                "data": [
                    {
                        "quantity": 1,
                        "price": {
                            "id": "price_test",
                            "unit_amount": 2000,
                            "currency": "usd",
                            "recurring": {"interval": "month", "interval_count": 1},
                        },
                    }
                ]
            },
        }
    )
    monkeypatch.setattr(checkout, "owned_subscription", AsyncMock(return_value=sub))
    monkeypatch.setattr(
        checkout, "conversion_trial", AsyncMock(return_value=(enrollment, sub))
    )
    monkeypatch.setattr(checkout, "quote_terms", AsyncMock(return_value=attempt.terms))
    monkeypatch.setattr(
        checkout, "reconcile_paid_activation", AsyncMock(return_value=True)
    )

    async def lost_response(*args, **kwargs):
        persisted = await get_attempt(enrollment.user_id)
        assert persisted and persisted.confirmed_at
        sub.status = "active"
        sub.latest_invoice = BillingInvoice(
            id="in_confirmed", customer=sub.customer, status="paid"
        )
        raise stripe.APIConnectionError("Payment settled but response was lost")

    mutation = AsyncMock(side_effect=lost_response)
    monkeypatch.setattr(checkout, "stripe_call", mutation)
    body = ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token)

    async def confirm_once():
        try:
            response = await checkout.confirm_activation(
                enrollment.user_id, attempt.id, body
            )
            return response.status
        except SubscriptionCheckoutUnavailable:
            return "concurrent_confirmation"

    results = await asyncio.gather(*[confirm_once() for _ in range(4)])
    assert set(results) <= {"processing", "ready", "concurrent_confirmation"}
    recovered = await checkout.get_activation(enrollment.user_id, attempt.id)
    assert recovered.status == "ready"
    assert recovered.return_to == "/chat/return-here"
    assert (
        await checkout.confirm_activation(enrollment.user_id, attempt.id, body)
    ).status == "ready"
    mutation.assert_awaited_once()
