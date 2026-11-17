"""A subscription cancelled because its account was deleted ends quietly.

The account deletion sends its own confirmation; the Stripe webhooks its
immediate cancel sets off must not add a "subscription ended" or "trial ended"
email, move the person in MailerLite, or record a person-level churn event
for a PostHog person the erasure deletes.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import credit
from backend.data.notifications import NotificationResult, SubscriptionPlan
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.notifications import lifecycle
from backend.notifications import trial as notices

CUSTOMER = "cus_1"
USER = SimpleNamespace(
    id="user-1",
    email="sam@example.com",
    name="Sam Carter",
    welcome_email_sent_at=None,
    marketing_opt_out_at=None,
    timezone=None,
)

PLAN = SubscriptionPlan(
    name="Pro",
    cycle="monthly",
    cycle_noun="month",
    label="Pro — monthly",
    price_display="$50.00 / month",
)


def _details(deleted: bool) -> dict:
    details = {"reason": "cancellation_requested"}
    if deleted:
        details["comment"] = credit.ACCOUNT_DELETION_CANCELLATION_COMMENT
    return details


def _paid(deleted: bool, **over) -> dict:
    sub = {
        "id": "sub_paid",
        "customer": CUSTOMER,
        "status": "canceled",
        "ended_at": 1789200000,
        "current_period_end": 1789200000,
        "cancel_at_period_end": False,
        "metadata": {},
        "items": {"data": [{"price": {"id": "price_1"}}]},
        "cancellation_details": _details(deleted),
    }
    sub.update(over)
    return sub


def _paid_mocks() -> dict:
    return {
        "user": AsyncMock(return_value=USER),
        "claim": AsyncMock(return_value=True),
        "notify": AsyncMock(return_value=NotificationResult(success=True)),
        "audience": AsyncMock(return_value=NotificationResult(success=True)),
        "ended": MagicMock(),
        "trial_notice": AsyncMock(return_value=False),
        "trial_update": AsyncMock(return_value=False),
        "plan": AsyncMock(return_value=PLAN),
    }


async def _run_paid(handler, mocks: dict) -> None:
    with (
        patch.object(lifecycle, "_user_for", mocks["user"]),
        patch.object(lifecycle, "claim_once", mocks["claim"]),
        patch.object(lifecycle, "queue_notification_async", mocks["notify"]),
        patch.object(lifecycle, "queue_audience_change", mocks["audience"]),
        patch.object(lifecycle, "track_subscription_ended", mocks["ended"]),
        patch.object(lifecycle, "notify_trial", mocks["trial_notice"]),
        patch.object(lifecycle, "on_trial_subscription_updated", mocks["trial_update"]),
        patch.object(lifecycle, "plan_from_subscription", mocks["plan"]),
        patch.object(
            lifecycle,
            "tier_and_cycle_from_subscription",
            AsyncMock(return_value=("PRO", "monthly")),
        ),
    ):
        await handler()


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [False, True])
async def test_a_deleted_accounts_paid_plan_ends_without_email_or_audience(deleted):
    mocks = _paid_mocks()
    await _run_paid(lambda: lifecycle.on_subscription_deleted(_paid(deleted)), mocks)

    if not deleted:
        # The same payload without the comment still churns as before.
        mocks["notify"].assert_awaited_once()
        mocks["audience"].assert_awaited_once()
        mocks["ended"].assert_called_once()
        return
    mocks["notify"].assert_not_awaited()
    mocks["audience"].assert_not_awaited()
    mocks["ended"].assert_not_called()
    mocks["claim"].assert_not_awaited()
    mocks["trial_notice"].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelling", [True, False])
async def test_a_deleted_accounts_cancel_flag_flip_sends_nothing(cancelling):
    """Stripe should only send ``deleted`` for an immediate cancel, but a
    cancelled/resumed email or field write must never reach a closed account."""
    mocks = _paid_mocks()
    subscription = _paid(True, cancel_at_period_end=cancelling, canceled_at=1789100000)
    previous = {"cancel_at_period_end": not cancelling, "canceled_at": 1789100000}
    await _run_paid(
        lambda: lifecycle.on_subscription_updated(subscription, previous), mocks
    )

    mocks["notify"].assert_not_awaited()
    mocks["audience"].assert_not_awaited()
    mocks["claim"].assert_not_awaited()
    mocks["trial_update"].assert_not_awaited()


# ── trials ──────────────────────────────────────────────────────────────────


def _trial() -> TrialState:
    now = datetime.now(UTC)
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id=CUSTOMER,
        offer=AcceptedTrialOffer(
            version="offer-v1",
            new_users_from=now - timedelta(days=1),
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
        checkout_session_id="cs_1",
        subscription_id="sub_trial",
        checkout_attempt=0,
        success_url="https://example.com",
        cancel_url="https://example.com",
        checkout_metadata={},
        status="canceled",
        card_verified_at=None,
        started_at=now - timedelta(days=2),
        ends_at=now,
        consumed_at=now - timedelta(days=2),
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


def _ended_trial(deleted: bool) -> dict:
    return {
        "id": "sub_trial",
        "customer": CUSTOMER,
        "status": "canceled",
        "trial_end": int(datetime.now(UTC).timestamp()),
        "metadata": {
            "trial_enrollment_id": "trial-1",
            "user_id": "user-1",
            "trial_checkout_attempt": "0",
        },
        "cancellation_details": _details(deleted),
    }


def _stripe_with_no_other_plan(subscription: dict) -> AsyncMock:
    """Stripe returns the trial itself, and no other live plan when listed."""

    async def call(fn, *args, **kwargs):
        if fn == notices.stripe.Subscription.list_async:
            return SimpleNamespace(data=[], has_more=False)
        return subscription

    return AsyncMock(side_effect=call)


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [False, True])
async def test_a_deleted_accounts_trial_ends_without_notice_or_audience(deleted):
    subscription = _ended_trial(deleted)
    notice = AsyncMock(return_value=NotificationResult(success=True))
    trial_audience = AsyncMock(return_value=NotificationResult(success=True))
    track = MagicMock()
    paid_email = AsyncMock()
    stripe_api = _stripe_with_no_other_plan(subscription)
    with (
        patch.object(notices, "stripe_call", stripe_api),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=_trial())
            ),
        ),
        patch.object(
            notices,
            "user_db",
            return_value=MagicMock(get_user_by_id=AsyncMock(return_value=USER)),
        ),
        patch.object(notices, "claim_once", AsyncMock(return_value=True)),
        patch.object(notices, "release_claim", AsyncMock()),
        patch.object(notices, "queue_notification_async", notice),
        patch(
            "backend.notifications.trial_audience.queue_audience_change",
            trial_audience,
        ),
        patch.object(notices, "_track_billing_event", track),
        patch.object(lifecycle, "_user_for", AsyncMock(return_value=USER)),
        patch.object(lifecycle, "queue_notification_async", paid_email),
        patch.object(lifecycle, "queue_audience_change", paid_email),
    ):
        await lifecycle.on_subscription_deleted(subscription)

    paid_email.assert_not_awaited()
    if not deleted:
        # Without the comment the trial's "ended" notice goes out as before.
        assert notice.await_args.args[0].data.kind == "ended"
        assert trial_audience.await_args.args[0].action.value == "remove_trial"
        assert track.call_args.args[0] == "trial_ended"
        return
    stripe_api.assert_not_awaited()
    notice.assert_not_awaited()
    trial_audience.assert_not_awaited()
    track.assert_not_called()
