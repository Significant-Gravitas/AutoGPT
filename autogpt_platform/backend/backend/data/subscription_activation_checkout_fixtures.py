"""Shared Stripe-boundary fixtures for activation recovery tests."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_stripe as billing
from backend.data.subscription_activation_models import (
    ActivationAttempt,
    ActivationTerms,
)


@pytest.fixture
def attempt():
    return ActivationAttempt(
        id="operation-1",
        user_id="user-1",
        subscription_id="sub_1",
        customer_id="cus_1",
        return_to="/chat/thread-123?resume=true",
        confirmed_at=None,
        terms=ActivationTerms(
            price_id="price_pro",
            accepted_offer_token="a" * 64,
            amount_due=4200,
            currency="usd",
            billing_interval="month",
            renewal_unit_amount=5000,
            renewal_terms="Renews every month; cancel before renewal.",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        ),
    )


@pytest.fixture
def live_subscription():
    return billing.BillingSubscription.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "trialing",
            "trial_end": int((datetime.now(UTC) + timedelta(days=1)).timestamp()),
            "metadata": {"user_id": "user-1", "trial_enrollment_id": "trial-1"},
            "items": {
                "data": [
                    {
                        "price": {
                            "id": "price_pro",
                            "unit_amount": 5000,
                            "currency": "usd",
                            "recurring": {"interval": "month", "interval_count": 1},
                        },
                        "quantity": 1,
                    }
                ]
            },
        }
    )


@pytest.fixture
def boundaries(monkeypatch, attempt, live_subscription, trial):
    @asynccontextmanager
    async def lock(user_id):
        yield

    confirmed = attempt.model_copy(update={"confirmed_at": datetime.now(UTC)})
    mocks = {
        "subscription_checkout_lock": lock,
        "get_attempt": AsyncMock(return_value=attempt),
        "save_confirmation": AsyncMock(return_value=confirmed),
        "save_quote": AsyncMock(return_value=attempt),
        "conversion_trial": AsyncMock(return_value=(trial, live_subscription)),
        "quote_terms": AsyncMock(return_value=attempt.terms),
        "owned_subscription": AsyncMock(return_value=live_subscription),
        "reconcile_paid_activation": AsyncMock(return_value=None),
        "stripe_call": AsyncMock(),
    }
    for name, value in mocks.items():
        monkeypatch.setattr(checkout, name, value)
    monkeypatch.setattr(
        billing, "get_subscription_trial", AsyncMock(return_value=trial)
    )
    monkeypatch.setattr(
        "backend.data.credit.build_price_to_tier_map",
        AsyncMock(return_value={"price_pro": SubscriptionTier.PRO}),
    )
    return SimpleNamespace(**mocks)


@pytest.fixture
def max_attempt(attempt, boundaries, trial, live_subscription):
    trial.subscription_id = live_subscription.id
    trial.consumed_at = datetime.now(UTC)
    attempt.terms = attempt.terms.model_copy(
        update={
            "plan": "MAX",
            "price_id": "price_max",
            "accepted_offer_token": trial.offer.token,
        }
    )
    live_subscription.items.data[0] = billing.ActivationItem.model_validate(
        {
            **live_subscription.items.data[0].model_dump(),
            "id": "si_owned",
        }
    )
    boundaries.quote_terms.return_value = attempt.terms
    boundaries.save_confirmation.return_value = attempt.model_copy(
        update={"confirmed_at": datetime.now(UTC)}
    )
    return attempt
