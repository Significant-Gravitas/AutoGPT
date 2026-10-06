"""Stripe API-version compatibility and commercial-term disclosure."""

from unittest.mock import AsyncMock

import pytest

from backend.data import subscription_activation_stripe as billing

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.mark.asyncio
@pytest.mark.parametrize("expanded", [False, True])
async def test_basil_invoice_payment_finds_default_intent(monkeypatch, expanded):
    invoice = {
        "id": "in_1",
        "customer": "cus_1",
        "status": "open",
        "payments": {
            "has_more": False,
            "data": [
                {
                    "invoice": "in_1",
                    "is_default": True,
                    "payment": {"type": "payment_intent", "payment_intent": "pi_1"},
                }
            ],
        },
    }
    retrieve = AsyncMock(return_value=invoice)
    monkeypatch.setattr(billing, "stripe_call", retrieve)
    initial = billing.BillingInvoice.model_validate(
        invoice
        if expanded
        else {
            "id": "in_1",
            "customer": "cus_1",
            "status": "open",
        }
    )
    assert await billing.invoice_payment_intent(initial) == "pi_1"
    assert retrieve.await_count == (0 if expanded else 1)


@pytest.mark.asyncio
async def test_acacia_invoice_payment_keeps_legacy_intent(monkeypatch):
    retrieve = AsyncMock()
    monkeypatch.setattr(billing, "stripe_call", retrieve)
    invoice = billing.BillingInvoice(
        id="in_1", customer="cus_1", payment_intent="pi_old"
    )
    assert await billing.invoice_payment_intent(invoice) == "pi_old"
    retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_discount_duration_and_tax_changes_require_new_consent(
    trial, subscription, monkeypatch
):
    subscription["items"]["data"][0]["price"].update(
        unit_amount=trial.offer.unit_amount,
        currency=trial.offer.currency,
        recurring={"interval": "month", "interval_count": 1},
    )
    live_subscription = billing.BillingSubscription.model_validate(subscription)
    invoice = {
        "customer": "cus_1",
        "currency": "usd",
        "amount_due": 2500,
        "discounts": [{"coupon": {"percent_off": 50, "duration": "forever"}}],
    }
    monkeypatch.setattr(billing, "stripe_call", AsyncMock(return_value=invoice))
    forever = await billing.quote_terms(trial, live_subscription)
    assert forever.renewal_discounts[0].percent_off == 50
    assert forever.renewal_discounts[0].duration == "forever"
    invoice["discounts"][0]["coupon"]["duration"] = "once"
    once = await billing.quote_terms(trial, live_subscription)
    assert once.amount_due == forever.amount_due
    assert not once.same_charge_as(forever)
    assert once.token != forever.token
    live_subscription.automatic_tax.enabled = True
    taxed = await billing.quote_terms(trial, live_subscription)
    assert not taxed.same_charge_as(once)
