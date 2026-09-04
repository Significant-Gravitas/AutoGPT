from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma import Prisma
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_stripe as fulfillment
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


@pytest.mark.asyncio
async def test_open_checkout_never_grants_or_consumes_trial(trial, session, boundaries):
    session["status"] = "open"
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["consumedAt"] is None
    assert saved["cardVerifiedAt"] is None
    assert saved["status"] == "checkout_pending"


@pytest.mark.asyncio
async def test_completed_checkout_consumes_trial_even_if_card_removed(
    trial, subscription, boundaries
):
    subscription["default_payment_method"] = None
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["consumedAt"] is not None
    assert saved["cardVerifiedAt"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"customer": "cus_other"},
        {"subscription": "sub_other"},
        {"metadata": {"user_id": "other", "trial_enrollment_id": "trial-1"}},
        {"payment_method_collection": "if_required"},
        {"payment_method_types": ["card", "us_bank_account"]},
    ],
)
async def test_checkout_proof_rejects_mismatched_identity_or_collection(
    trial, session, boundaries, changed
):
    session.update(changed)
    with pytest.raises(ValueError, match="Checkout"):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    boundaries.user.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversion_invoice_identity_is_immutable(trial, boundaries):
    now = datetime.now(UTC)
    invoice = fulfillment.Invoice(
        id="in_first",
        status="paid",
        created=int(now.timestamp()),
        billing_reason="subscription_cycle",
    )
    snapshot = fulfillment.SubscriptionSnapshot(
        id="sub_1",
        customer="cus_1",
        status="active",
        latest_invoice=invoice,
    )
    await fulfillment._save_snapshot(
        trial, snapshot, SubscriptionTier.PRO, now, boundaries, True
    )
    first = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert first["stripeConversionInvoiceId"] == "in_first"
    converted = trial.model_copy(
        update={"converted_at": now, "conversion_invoice_id": "in_first"}
    )
    invoice.id = "in_renewal"
    await fulfillment._save_snapshot(
        converted, snapshot, SubscriptionTier.PRO, now, boundaries, True
    )
    later = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert later["stripeConversionInvoiceId"] == "in_first"


@pytest.mark.asyncio
async def test_completed_card_checkout_grants_trial(trial, boundaries):
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items",
    [
        None,
        {"data": [], "has_more": False},
        {"data": [{"price": {"id": "price_other"}, "quantity": 1}], "has_more": False},
        {"data": [{"price": {"id": "price_pro"}, "quantity": 2}], "has_more": False},
        {"data": [{"price": {"id": "price_pro"}, "quantity": 1}], "has_more": True},
    ],
)
async def test_first_conversion_rejects_unaccepted_price_or_quantity(
    trial, subscription, boundaries, items
):
    end = int(datetime.now(UTC).timestamp()) - 60
    subscription.update(
        status="active",
        trial_end=end,
        items=items,
        latest_invoice={
            "id": "in_1",
            "status": "paid",
            "created": end,
            "billing_reason": "subscription_cycle",
        },
    )
    with pytest.raises(ValueError, match="accepted"):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    boundaries.user.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_attempt_is_acknowledged_without_rewriting_current_state(
    trial, boundaries
):
    trial.checkout_attempt = 1
    trial.subscription_id = "sub_current"
    boundaries.user.find_unique_or_raise = AsyncMock(
        return_value=MagicMock(subscriptionTier=SubscriptionTier.PRO)
    )
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.PRO
    boundaries.user.update_many.assert_not_awaited()
    boundaries.subscriptiontrial.update.assert_not_awaited()
