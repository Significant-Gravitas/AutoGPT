"""Payment review dates and tax come from the accepted subscription preview."""

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import AsyncMock

import pytest

from backend.data import subscription_activation_stripe as billing
from backend.data.subscription_activation_models import (
    ActivationTerms,
    ActivationUnavailable,
)

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.fixture
def quote_context(trial, subscription, monkeypatch):
    trial.ends_at = datetime(2026, 10, 12, 12, tzinfo=UTC)
    subscription["items"]["data"][0]["price"].update(
        unit_amount=trial.offer.unit_amount,
        currency=trial.offer.currency,
        recurring={"interval": "month", "interval_count": 1},
    )
    preview = {
        "customer": "cus_1",
        "currency": "usd",
        "amount_due": 6000,
        "total_tax_amounts": [{"amount": 700}, {"amount": 300}],
        "lines": {
            "has_more": False,
            "data": [
                {
                    "type": "subscription",
                    "subscription": "sub_1",
                    "price": {"id": "price_pro"},
                    "proration": False,
                    "period": {"start": 1791374400, "end": 1794052800},
                }
            ],
        },
    }
    call = AsyncMock(return_value=preview)
    monkeypatch.setattr(billing, "stripe_call", call)
    return trial, billing.BillingSubscription.model_validate(subscription), preview


@pytest.mark.asyncio
@pytest.mark.parametrize("api_version", ["acacia", "basil"])
async def test_quote_discloses_trial_end_paid_period_and_actual_tax(
    quote_context, api_version
):
    trial, subscription, preview = quote_context
    if api_version == "basil":
        preview["total_taxes"] = preview.pop("total_tax_amounts")
        line = preview["lines"]["data"][0]
        line.pop("type")
        line.pop("subscription")
        line.pop("price")
        line["parent"] = {
            "type": "subscription_item_details",
            "subscription_item_details": {"subscription": "sub_1", "proration": False},
        }
        line["pricing"] = {"price_details": {"price": "price_pro"}}
    terms = await billing.quote_terms(trial, subscription)
    assert terms.trial_ends_at == trial.ends_at
    assert terms.period_starts_at == datetime(2026, 10, 7, 12, tzinfo=UTC)
    assert terms.next_payment_at == datetime(2026, 11, 7, 12, tzinfo=UTC)
    assert terms.tax_amount == 1000
    assert terms.amount_due == 6000


@pytest.mark.asyncio
async def test_quote_does_not_invent_zero_tax(quote_context):
    trial, subscription, preview = quote_context
    preview.pop("total_tax_amounts")
    assert (await billing.quote_terms(trial, subscription)).tax_amount is None
    preview["total_taxes"] = []
    assert (await billing.quote_terms(trial, subscription)).tax_amount == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unrelated", "missing", "incomplete"])
async def test_quote_requires_the_owned_subscription_paid_period(
    quote_context, invalid
):
    trial, subscription, preview = quote_context
    if invalid == "unrelated":
        preview["lines"]["data"][0]["subscription"] = "sub_other"
    elif invalid == "missing":
        preview["lines"]["data"] = []
    else:
        preview["lines"]["has_more"] = True
    with pytest.raises(ActivationUnavailable, match="paid period"):
        await billing.quote_terms(trial, subscription)


@pytest.mark.asyncio
async def test_quote_time_moves_without_requiring_new_charge_consent(quote_context):
    trial, subscription, preview = quote_context
    original = await billing.quote_terms(trial, subscription)
    preview["lines"]["data"][0]["period"]["start"] += 60
    preview["lines"]["data"][0]["period"]["end"] += 60
    updated = await billing.quote_terms(trial, subscription)
    assert original.same_charge_as(updated)
    preview["total_tax_amounts"][0]["amount"] += 1
    assert not original.same_charge_as(await billing.quote_terms(trial, subscription))


def test_previously_persisted_quote_retains_its_confirmation_token():
    terms = ActivationTerms(
        price_id="price_pro",
        accepted_offer_token="offer",
        amount_due=5000,
        currency="usd",
        billing_interval="month",
        renewal_unit_amount=5000,
        renewal_terms="Renews monthly.",
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )
    original = terms.model_dump_json(
        exclude={"trial_ends_at", "period_starts_at", "next_payment_at", "tax_amount"}
    )
    restored = ActivationTerms.model_validate_json(original)
    assert restored.token == sha256(original.encode()).hexdigest()
    assert restored.persisted_json() == original
