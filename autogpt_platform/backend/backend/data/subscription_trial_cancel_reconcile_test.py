"""Reconciling a trial whose cancellation is scheduled: with the flag on it
keeps access until the trial ends, then ends, or converts once resumed."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_stripe as fulfillment
from backend.data.subscription_trial_rejection import TrialRejectionReason
from backend.util.feature_flag import Flag

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.fixture(autouse=True)
def cancel_flag(trial_cancel_flag):
    return trial_cancel_flag


@pytest.mark.asyncio
async def test_scheduled_cancellation_keeps_access_until_trial_end(
    trial, subscription, boundaries, cancel_flag
):
    cancel_flag.return_value = (True, True)
    subscription["cancel_at_period_end"] = True
    with patch.object(
        fulfillment.stripe.Subscription, "cancel_async", AsyncMock()
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()
    cancel_flag.assert_awaited_once_with(
        Flag.TRIAL_CANCEL_AT_PERIOD_END, trial.user_id, default=False
    )
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["status"] == "trialing"
    assert saved["cancelAtPeriodEnd"] is True
    assert saved["notificationRevision"] == 1
    assert boundaries.user.update_many.await_args.kwargs["data"] == {
        "subscriptionTier": SubscriptionTier.TRIAL
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [(False, False), (True, False)])
async def test_unreadable_flag_keeps_a_scheduled_cancellations_access(
    trial, subscription, boundaries, cancel_flag, value
):
    """Only an authoritative "off" ends a trial early: that cannot be undone."""
    cancel_flag.return_value = value
    subscription["cancel_at_period_end"] = True
    with patch.object(
        fulfillment.stripe.Subscription, "cancel_async", AsyncMock()
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_recorded_cancellation_keeps_access_after_the_flag_turns_off(
    trial, subscription, boundaries, cancel_flag
):
    """Turning the flag off never takes back access a person was promised."""
    now = datetime.now(UTC)
    trial = trial.model_copy(
        update={
            "status": "trialing",
            "consumed_at": now,
            "card_verified_at": now,
            "subscription_id": "sub_1",
            "cancel_at_period_end": True,
            "notification_revision": 1,
        }
    )
    subscription["cancel_at_period_end"] = True
    with patch.object(
        fulfillment.stripe.Subscription, "cancel_async", AsyncMock()
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()
    cancel_flag.assert_not_awaited()
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["notificationRevision"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,claimed,fingerprint",
    [
        (TrialRejectionReason.INTRO_OFFER_ALREADY_USED, False, "fp_test"),
        (TrialRejectionReason.CARD_VERIFICATION_FAILED, True, None),
    ],
)
async def test_rejected_trial_is_still_canceled_immediately_with_the_flag_on(
    trial, subscription, boundaries, cancel_flag, reason, claimed, fingerprint
):
    cancel_flag.return_value = (True, True)
    subscription["cancel_at_period_end"] = True
    subscription["default_payment_method"]["card"]["fingerprint"] = fingerprint
    canceled = {**subscription, "status": "canceled"}
    with (
        patch.object(
            fulfillment, "claim_trial_identities", AsyncMock(return_value=claimed)
        ),
        patch.object(
            fulfillment.stripe.Subscription,
            "cancel_async",
            AsyncMock(return_value=canceled),
        ) as cancel,
    ):
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    assert cancel.await_args.kwargs["invoice_now"] is False
    assert cancel.await_args.kwargs["prorate"] is False
    assert cancel.await_args.kwargs["cancellation_details"] == {
        "comment": reason.stripe_comment
    }
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"]["rejectionReason"]
        == reason
    )
    cancel_flag.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_cancellation_ends_access_when_stripe_ends_the_trial(
    trial, subscription, boundaries
):
    """customer.subscription.deleted at trial end: no tier, no conversion."""
    now = datetime.now(UTC)
    trial = trial.model_copy(
        update={
            "status": "trialing",
            "consumed_at": now - timedelta(days=7),
            "card_verified_at": now - timedelta(days=7),
            "subscription_id": "sub_1",
            "cancel_at_period_end": True,
            "notification_revision": 1,
        }
    )
    ended_at = int(now.timestamp()) - 30
    subscription.update(
        status="canceled",
        cancel_at_period_end=True,
        trial_end=ended_at,
        ended_at=ended_at,
    )
    with patch.object(
        fulfillment.stripe.Subscription, "cancel_async", AsyncMock()
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    cancel.assert_not_awaited()
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["status"] == "canceled"
    assert saved["endsAt"] == datetime.fromtimestamp(ended_at, UTC)
    assert saved["convertedAt"] is None
    assert saved["stripeConversionInvoiceId"] is None
    assert saved["notificationRevision"] == 1
    assert boundaries.user.update_many.await_args.kwargs["data"] == {
        "subscriptionTier": SubscriptionTier.NO_TIER
    }


@pytest.mark.asyncio
async def test_resumed_trial_converts_at_trial_end(trial, subscription, boundaries):
    now = datetime.now(UTC)
    trial = trial.model_copy(
        update={
            "status": "trialing",
            "consumed_at": now - timedelta(days=7),
            "card_verified_at": now - timedelta(days=7),
            "subscription_id": "sub_1",
            "notification_revision": 2,
        }
    )
    end = int(now.timestamp()) - 60
    subscription.update(
        status="active",
        trial_end=end,
        latest_invoice={
            "id": "in_1",
            "status": "paid",
            "created": end,
            "billing_reason": "subscription_cycle",
        },
    )
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.PRO
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["convertedAt"] is not None
    assert saved["stripeConversionInvoiceId"] == "in_1"
    assert saved["notificationRevision"] == 2
