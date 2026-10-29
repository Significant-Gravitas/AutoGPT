"""Replay actual Stripe test-clock states through the entitlement decision."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from prisma.enums import SubscriptionTier
from pydantic import BaseModel

from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_stripe import (
    SubscriptionSnapshot,
    trial_subscription_tier,
)


class CapturedScenario(BaseModel):
    before: SubscriptionSnapshot
    after: SubscriptionSnapshot


class CapturedSimulation(BaseModel):
    start_time: int
    end_time: int
    scenarios: dict[str, CapturedScenario]


CAPTURE = CapturedSimulation.model_validate_json(
    (Path(__file__).parent / "test_data" / "trial_stripe_simulation.json").read_text()
)


@pytest.mark.parametrize("scenario", ["success", "failure", "cancel"])
def test_real_zero_dollar_initial_invoice_is_trial_not_paid_access(scenario: str):
    snapshot = CAPTURE.scenarios[scenario].before
    now = datetime.fromtimestamp(CAPTURE.start_time, UTC)
    assert snapshot.latest_invoice is not None
    assert snapshot.latest_invoice.status == "paid"
    assert snapshot.latest_invoice.billing_reason == "subscription_create"
    assert (
        trial_subscription_tier(enrollment(snapshot), snapshot, now)
        == SubscriptionTier.TRIAL
    )


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("success", SubscriptionTier.PRO),
        ("failure", SubscriptionTier.NO_TIER),
        ("cancel", SubscriptionTier.NO_TIER),
    ],
)
def test_real_trial_conversion_states(scenario: str, expected: SubscriptionTier):
    snapshot = CAPTURE.scenarios[scenario].after
    now = datetime.fromtimestamp(CAPTURE.end_time, UTC)
    assert trial_subscription_tier(enrollment(snapshot), snapshot, now) == expected


def enrollment(snapshot: SubscriptionSnapshot) -> TrialState:
    start = datetime.fromtimestamp(CAPTURE.start_time, UTC)
    return TrialState(
        id="captured-trial",
        user_id="captured-user",
        customer_id=snapshot.customer,
        offer=AcceptedTrialOffer(
            version="stripe-simulation-not-launch-offer",
            new_users_from=start,
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_test_1",
            unit_amount=5000,
            currency="usd",
        ),
        checkout_session_id=None,
        subscription_id=snapshot.id,
        checkout_attempt=0,
        success_url="https://example.com/ok",
        cancel_url="https://example.com/no",
        checkout_metadata={},
        status="trialing",
        card_verified_at=start,
        started_at=start,
        ends_at=datetime.fromtimestamp(CAPTURE.start_time + 7 * 86400, UTC),
        consumed_at=start,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )
