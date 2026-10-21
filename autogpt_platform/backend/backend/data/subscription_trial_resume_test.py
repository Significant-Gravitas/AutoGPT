from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_checkout as checkout
from backend.data import subscription_trial_eligibility as eligibility
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer


@pytest.fixture
def trial():
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="resume-v1",
            new_users_from=datetime(2026, 9, 1, tzinfo=UTC),
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
        subscription_id="sub_1",
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
def session(trial):
    return stripe.checkout.Session.construct_from(
        {
            "id": "cs_1",
            "status": "open",
            "mode": "subscription",
            "url": "https://checkout.stripe.com/test",
            "customer": trial.customer_id,
            "subscription": "sub_1",
            "metadata": {
                "trial_enrollment_id": trial.id,
                "user_id": trial.user_id,
                "trial_checkout_attempt": "0",
                "trial_offer_version": trial.offer.version,
            },
        },
        "test-key",
    )


@pytest.fixture
def subscriptions(session):
    return [{"id": "sub_1", "status": "trialing", "metadata": dict(session.metadata)}]


@pytest.fixture
def boundaries(trial, session, subscriptions):
    user = MagicMock(
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
        subscription_tier=SubscriptionTier.NO_TIER,
    )
    with (
        patch.object(checkout, "_find_checkout", AsyncMock(return_value=session)),
        patch.object(checkout, "expire_other_subscription_checkouts", AsyncMock()),
        patch.object(eligibility, "get_user_by_id", AsyncMock(return_value=user)),
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(
                side_effect=lambda *args, **kwargs: stripe.ListObject.construct_from(
                    {"data": subscriptions, "has_more": False}, "test-key"
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
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["trialing", "incomplete"])
async def test_pending_checkout_can_resume_its_own_unfinished_subscription(
    trial, subscriptions, boundaries, status
):
    subscriptions[0]["status"] = status
    assert await checkout._resume_checkout(trial) == "https://checkout.stripe.com/test"


@pytest.mark.asyncio
async def test_another_subscription_still_disqualifies_trial(
    trial, subscriptions, boundaries
):
    subscriptions.append({"id": "sub_previous", "status": "canceled", "metadata": {}})
    with pytest.raises(checkout.TrialUnavailable, match="not eligible"):
        await checkout._resume_checkout(trial)


@pytest.mark.asyncio
async def test_paid_subscription_cannot_be_ignored_as_unfinished_checkout(
    trial, subscriptions, boundaries
):
    subscriptions[0]["status"] = "active"
    with pytest.raises(checkout.TrialUnavailable, match="not eligible"):
        await checkout._resume_checkout(trial)
