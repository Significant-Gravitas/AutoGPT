from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma import Prisma

from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer


@pytest.fixture
def trial():
    now = datetime.now(UTC)
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="fulfillment-v1",
            new_users_from=now - timedelta(days=1),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=5000,
            currency="usd",
        ),
        checkout_session_id="cs_1",
        subscription_id=None,
        checkout_attempt=0,
        success_url="https://example.com/ok",
        cancel_url="https://example.com/no",
        checkout_metadata={},
        status="checkout_pending",
        card_verified_at=None,
        started_at=None,
        ends_at=None,
        consumed_at=None,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


@pytest.fixture
def subscription(trial):
    now = datetime.now(UTC)
    return {
        "id": "sub_1",
        "customer": trial.customer_id,
        "status": "trialing",
        "metadata": {
            "user_id": trial.user_id,
            "trial_enrollment_id": trial.id,
            "trial_offer_version": trial.offer.version,
            "trial_checkout_attempt": "0",
        },
        "trial_start": int(now.timestamp()),
        "trial_end": int((now + timedelta(days=7)).timestamp()),
        "default_payment_method": {
            "id": "pm_1",
            "type": "card",
            "card": {"exp_month": 12, "exp_year": 2030},
        },
        "pending_setup_intent": None,
        "items": {
            "data": [{"price": {"id": trial.offer.price_id}, "quantity": 1}],
            "has_more": False,
        },
    }


@pytest.fixture
def session(trial):
    return {
        "id": "cs_1",
        "status": "complete",
        "mode": "subscription",
        "customer": trial.customer_id,
        "subscription": "sub_1",
        "payment_method_types": ["card"],
        "payment_method_collection": "always",
        "metadata": {
            "user_id": trial.user_id,
            "trial_enrollment_id": trial.id,
            "trial_offer_version": trial.offer.version,
            "trial_checkout_attempt": "0",
        },
    }


@pytest.fixture
def boundaries(subscription, session):
    tx = MagicMock(spec=Prisma)
    tx.subscriptiontrial = MagicMock(update=AsyncMock())
    tx.user = MagicMock(update_many=AsyncMock())
    with (
        patch.object(
            stripe.Customer,
            "retrieve_async",
            AsyncMock(
                return_value={
                    "id": "cus_1",
                    "invoice_settings": {"default_payment_method": None},
                }
            ),
        ),
        patch.object(
            stripe.Subscription, "retrieve_async", AsyncMock(return_value=subscription)
        ),
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(return_value=MagicMock(data=[])),
        ),
        patch.object(
            stripe.checkout.Session,
            "retrieve_async",
            AsyncMock(
                side_effect=lambda *args, **kwargs: stripe.checkout.Session.construct_from(
                    session, "test-key"
                )
            ),
        ),
        patch.object(
            stripe.checkout.Session,
            "list_async",
            AsyncMock(
                return_value=stripe.ListObject.construct_from(
                    {"data": [], "has_more": False}, "test-key"
                )
            ),
        ),
    ):
        yield tx
