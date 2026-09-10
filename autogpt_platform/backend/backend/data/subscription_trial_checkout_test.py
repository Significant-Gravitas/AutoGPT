from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_checkout as checkout
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_stripe import (
    SubscriptionSnapshot,
    trial_subscription_tier,
)


@pytest.fixture
def checkout_guard():
    with patch.object(checkout, "subscription_checkout_lock", return_value=AsyncMock()):
        yield


@pytest.fixture
def trial() -> TrialState:
    return TrialState(
        id="trial-1",
        user_id="user-1",
        offer=AcceptedTrialOffer(
            version="offer-a",
            new_users_from=datetime(2026, 9, 10, tzinfo=UTC),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=2000,
            currency="usd",
        ),
        customer_id="cus_1",
        checkout_session_id=None,
        subscription_id=None,
        checkout_attempt=0,
        success_url="https://platform.agpt.co/settings/billing?trial=success",
        cancel_url="https://platform.agpt.co/settings/billing",
        checkout_metadata={"datafast_visitor_id": "visitor-1"},
        status="checkout_pending",
        card_verified_at=None,
        started_at=None,
        ends_at=None,
        consumed_at=None,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


def test_card_required_and_cancel_if_card_disappears(trial):
    params = checkout.trial_checkout_params(trial)
    assert params["mode"] == "subscription"
    assert params["payment_method_types"] == ["card"]
    assert params["payment_method_collection"] == "always"
    assert params["subscription_data"]["trial_period_days"] == 7
    assert params["subscription_data"]["trial_settings"] == {
        "end_behavior": {"missing_payment_method": "cancel"}
    }
    assert params["allow_promotion_codes"] is False


def test_uses_accepted_price_and_preserves_attribution(trial):
    params = checkout.trial_checkout_params(trial)
    assert params["line_items"] == [{"price": "price_pro", "quantity": 1}]
    assert params["metadata"]["datafast_visitor_id"] == "visitor-1"
    assert params["subscription_data"]["metadata"] == params["metadata"]
    assert params["metadata"]["trial_enrollment_id"] == trial.id
    assert params["metadata"]["trial_offer_version"] == trial.offer.version


def test_attribution_cannot_overwrite_server_owned_identity(trial):
    trial.checkout_metadata.update(user_id="other-user", trial_enrollment_id="other")
    params = checkout.trial_checkout_params(trial)
    assert params["metadata"]["user_id"] == trial.user_id
    assert params["metadata"]["trial_enrollment_id"] == trial.id


@pytest.mark.parametrize(
    "change",
    [
        {"duration_days": 14},
        {"price_id": "price_other"},
        {"unit_amount": 3000},
        {"total_cost_limit": 2_000_000},
    ],
)
def test_terms_token_changes_even_if_operator_reuses_offer_version(trial, change):
    changed = AcceptedTrialOffer.model_validate({**trial.offer.model_dump(), **change})
    assert changed.token != trial.offer.token


@pytest.mark.asyncio
async def test_disabled_offer_does_not_create_stripe_customer_or_session(
    trial, checkout_guard
):
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=None)),
        patch.object(checkout, "get_stripe_customer_id", AsyncMock()) as customer,
        patch.object(checkout, "_resume_checkout", AsyncMock()) as resume,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="not available"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
            )
    customer.assert_not_awaited()
    resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_previously_consumed_trial_is_never_reopened(trial, checkout_guard):
    trial.consumed_at = datetime(2026, 9, 10, tzinfo=UTC)
    trial.status = "canceled"
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=trial.offer)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(checkout, "_resume_checkout", AsyncMock()) as resume,
    ):
        with pytest.raises(checkout.TrialUnavailable, match="already been used"):
            await checkout.create_trial_checkout(
                trial.user_id,
                trial.offer.token,
                trial.success_url,
                trial.cancel_url,
                {},
            )
    resume.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_checkout_reuses_accepted_offer(trial, checkout_guard):
    future_offer = trial.offer.model_copy(update={"duration_days": 14})
    with (
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=future_offer)),
        patch.object(checkout, "get_subscription_trial", AsyncMock(return_value=trial)),
        patch.object(
            checkout, "_resume_checkout", AsyncMock(return_value="checkout")
        ) as resume,
    ):
        result = await checkout.create_trial_checkout(
            trial.user_id, trial.offer.token, trial.success_url, trial.cancel_url, {}
        )
    assert result == "checkout"
    resume.assert_awaited_once_with(trial)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, SubscriptionTier.TRIAL),
        ({"cancel_at_period_end": True}, SubscriptionTier.TRIAL),
        ({"default_payment_method": None}, SubscriptionTier.NO_TIER),
        ({"pending_setup_intent": "seti_needs_auth"}, SubscriptionTier.NO_TIER),
        ({"status": "past_due"}, SubscriptionTier.NO_TIER),
        ({"status": "canceled"}, SubscriptionTier.NO_TIER),
        ({"trial_end": 0}, SubscriptionTier.NO_TIER),
        ({"status": "active"}, SubscriptionTier.NO_TIER),
    ],
)
def test_entitlement_requires_completed_card_setup_and_unexpired_trial(
    trial, overrides, expected
):
    now = datetime(2026, 9, 10, tzinfo=UTC)
    subscription = SubscriptionSnapshot.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "trialing",
            "trial_end": int(now.timestamp()) + 86400,
            "default_payment_method": {
                "id": "pm_1",
                "type": "card",
                "card": {"exp_month": 12, "exp_year": 2030},
            },
            **overrides,
        }
    )
    assert trial_subscription_tier(trial, subscription, now) == expected


@pytest.mark.parametrize(
    "invoice_status,reason,created_delta,expected",
    [
        ("paid", "subscription_cycle", 0, SubscriptionTier.PRO),
        ("open", "subscription_cycle", 0, SubscriptionTier.NO_TIER),
        ("draft", "subscription_cycle", 0, SubscriptionTier.NO_TIER),
        ("uncollectible", "subscription_cycle", 0, SubscriptionTier.NO_TIER),
        ("paid", "subscription_create", 0, SubscriptionTier.NO_TIER),
        ("paid", "subscription_cycle", -1, SubscriptionTier.NO_TIER),
    ],
)
def test_paid_conversion_requires_paid_post_trial_invoice(
    trial, invoice_status, reason, created_delta, expected
):
    now = datetime(2026, 9, 10, tzinfo=UTC)
    end = int(now.timestamp())
    subscription = SubscriptionSnapshot.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "active",
            "trial_end": end,
            "latest_invoice": {
                "id": "in_1",
                "status": invoice_status,
                "billing_reason": reason,
                "created": end + created_delta,
            },
        }
    )
    assert trial_subscription_tier(trial, subscription, now) == expected
