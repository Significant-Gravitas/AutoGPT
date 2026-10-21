import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial, SubscriptionTrialClaim, User

from backend.data import db
from backend.data import subscription_trial_checkout as checkout
from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.subscription_trial import (
    TrialState,
    get_subscription_trial,
    reserve_subscription_trial,
)
from backend.data.subscription_trial_claims import claim_trial_identities
from backend.data.subscription_trial_config import AcceptedTrialOffer, TrialOffer
from backend.data.subscription_trial_rejection import TrialRejectionReason
from backend.data.subscription_trial_stripe import reconcile_trial_subscription

pytestmark = fixtures.pytestmark


@pytest.mark.asyncio
async def test_declined_hold_leaves_the_trial_for_a_retry_with_another_card(
    started, stripe_trial
):
    stripe_trial.hold.side_effect = stripe.CardError(
        "Your card has insufficient funds.",
        None,
        "card_declined",
        json_body={
            "error": {"code": "card_declined", "decline_code": "insufficient_funds"}
        },
    )
    result = await reconcile_trial_subscription(started.user_id, "sub_declined")
    assert result is not None and result[1] == SubscriptionTier.NO_TIER

    trial = await get_subscription_trial(started.user_id)
    assert trial is not None and trial.consumed_at is None
    assert trial.status == "checkout_pending" and trial.checkout_attempt == 1
    assert trial.rejection_reason == TrialRejectionReason.CARD_DECLINED
    async with db.transaction() as tx:
        assert not await claim_trial_identities(
            trial.model_copy(
                update={
                    "id": str(uuid4()),
                    "user_id": str(uuid4()),
                    "customer_id": f"cus_{uuid4().hex}",
                }
            ),
            stripe_trial.fingerprint,
            tx,
        ), "a declined card still counts toward one trial per card"

    url = await checkout.create_trial_checkout(
        started.user_id,
        trial.offer.token,
        "https://example.com/ok",
        "https://example.com/no",
        {},
    )
    assert url == "https://checkout.stripe.test/retry"
    retry = stripe_trial.create_session.await_args.kwargs
    assert retry["idempotency_key"] == f"trial-checkout-{trial.id}-1"
    assert retry["metadata"]["trial_checkout_attempt"] == "1"


@pytest.mark.asyncio
async def test_duplicate_webhooks_place_one_hold(started, stripe_trial):
    results = await asyncio.gather(
        reconcile_trial_subscription(started.user_id, "sub_declined"),
        reconcile_trial_subscription(started.user_id, "sub_declined"),
    )
    assert [result[1] for result in results if result] == [SubscriptionTier.TRIAL] * 2
    stripe_trial.hold.assert_awaited_once()
    stripe_trial.release.assert_awaited_once()
    trial = await get_subscription_trial(started.user_id)
    assert trial is not None and trial.consumed_at is not None and trial.active


@pytest_asyncio.fixture
async def started(enrollment_offer):
    fixtures.require_disposable_database()
    owns_connection = not db.is_connected()
    await db.connect()
    user_id = str(uuid4())
    customer_id = f"cus_{uuid4().hex}"
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"{user_id}@example.com",
            "stripeCustomerId": customer_id,
        }
    )
    trial = await reserve_subscription_trial(
        user_id,
        enrollment_offer,
        customer_id,
        "https://example.com/ok",
        "https://example.com/no",
        {},
    )
    await SubscriptionTrial.prisma().update(
        where={"id": trial.id}, data={"stripeCheckoutSessionId": "cs_attempt_0"}
    )
    yield trial.model_copy(update={"checkout_session_id": "cs_attempt_0"})
    await SubscriptionTrialClaim.prisma().delete_many(where={"trialId": trial.id})
    await User.prisma().delete_many(where={"id": user_id})
    if owns_connection:
        await db.disconnect()


@pytest.fixture
def enrollment_offer() -> AcceptedTrialOffer:
    return AcceptedTrialOffer(
        version="hold-v1",
        new_users_from=datetime(2026, 9, 1, tzinfo=UTC),
        duration_days=7,
        tier="PRO",
        billing_cycle="monthly",
        daily_cost_limit=250_000,
        weekly_cost_limit=1_000_000,
        total_cost_limit=1_000_000,
        onboarding_credit_amount=0,
        price_id="price_hold",
        unit_amount=5000,
        currency="usd",
    )


@pytest.fixture
def stripe_trial(started: TrialState, enrollment_offer: AcceptedTrialOffer):
    """Stripe as it stands after the user completed the trial Checkout."""
    now = datetime.now(UTC)
    metadata = {
        "user_id": started.user_id,
        "trial_enrollment_id": started.id,
        "trial_offer_version": enrollment_offer.version,
        "trial_checkout_attempt": "0",
    }
    fingerprint = f"fp_{uuid4().hex}"
    subscription = {
        "id": "sub_declined",
        "customer": started.customer_id,
        "status": "trialing",
        "metadata": metadata,
        "trial_start": int(now.timestamp()),
        "trial_end": int((now + timedelta(days=7)).timestamp()),
        "default_payment_method": {
            "id": "pm_1",
            "type": "card",
            "card": {"exp_month": 12, "exp_year": 2030, "fingerprint": fingerprint},
        },
        "items": {
            "data": [{"price": {"id": enrollment_offer.price_id}, "quantity": 1}],
            "has_more": False,
        },
    }
    session = {
        "id": "cs_attempt_0",
        "object": "checkout.session",
        "status": "complete",
        "mode": "subscription",
        "customer": started.customer_id,
        "subscription": "sub_declined",
        "payment_method_types": ["card"],
        "payment_method_collection": "always",
        "metadata": metadata,
    }

    async def list_sessions(customer: str, limit: int, status: str | None = None):
        return _page(session if status in (None, session["status"]) else None)

    async def list_subscriptions(customer: str, status: str, limit: int):
        return _page(
            subscription if status in ("all", subscription["status"]) else None
        )

    async def cancel(subscription_id: str, **params):
        subscription.update(
            status="canceled",
            cancellation_details=params.get("cancellation_details"),
            ended_at=int(now.timestamp()),
        )
        return dict(subscription)

    hold = AsyncMock(return_value=_intent("requires_capture"))
    release = AsyncMock(return_value=_intent("canceled"))
    create_session = AsyncMock(
        return_value=stripe.checkout.Session.construct_from(
            {
                "id": "cs_attempt_1",
                "status": "open",
                "url": "https://checkout.stripe.test/retry",
            },
            "test-key",
        )
    )
    offer = TrialOffer(
        **enrollment_offer.model_dump(exclude={"price_id", "unit_amount", "currency"})
    )
    with (
        patch.object(
            stripe.Subscription,
            "retrieve_async",
            AsyncMock(side_effect=lambda *a, **k: dict(subscription)),
        ),
        patch.object(
            stripe.Subscription, "list_async", AsyncMock(side_effect=list_subscriptions)
        ),
        patch.object(
            stripe.Subscription, "cancel_async", AsyncMock(side_effect=cancel)
        ),
        patch.object(
            stripe.checkout.Session,
            "retrieve_async",
            AsyncMock(
                side_effect=lambda *a, **k: stripe.checkout.Session.construct_from(
                    session, "test-key"
                )
            ),
        ),
        patch.object(
            stripe.checkout.Session, "list_async", AsyncMock(side_effect=list_sessions)
        ),
        patch.object(stripe.checkout.Session, "create_async", create_session),
        patch.object(stripe.PaymentIntent, "create_async", hold),
        patch.object(stripe.PaymentIntent, "cancel_async", release),
        patch.object(checkout, "get_trial_offer", AsyncMock(return_value=offer)),
    ):
        yield SimpleNamespace(
            hold=hold,
            release=release,
            create_session=create_session,
            fingerprint=fingerprint,
        )


def _intent(status: str) -> stripe.PaymentIntent:
    return stripe.PaymentIntent.construct_from(
        {"id": "pi_hold", "object": "payment_intent", "status": status}, "test-key"
    )


def _page(item: dict | None) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {"data": [item] if item else [], "has_more": False}, "test-key"
    )
