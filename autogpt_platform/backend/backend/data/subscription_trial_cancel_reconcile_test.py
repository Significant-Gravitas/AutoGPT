"""Reconciling a trial whose cancellation is scheduled: it keeps access until
the trial ends, then ends, or converts once resumed."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_stripe as fulfillment
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_rejection import TrialRejectionReason

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


def _recorded_cancel_pending(trial: TrialState) -> TrialState:
    now = datetime.now(UTC)
    return trial.model_copy(
        update={
            "status": "trialing",
            "consumed_at": now - timedelta(days=2),
            "card_verified_at": now - timedelta(days=2),
            "subscription_id": "sub_1",
            "cancel_at_period_end": True,
            "notification_revision": 1,
        }
    )


@pytest.mark.asyncio
async def test_scheduled_cancellation_keeps_access_until_trial_end(
    trial, subscription, boundaries
):
    subscription["cancel_at_period_end"] = True
    with patch.object(
        fulfillment.stripe.Subscription, "cancel_async", AsyncMock()
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["status"] == "trialing"
    assert saved["cancelAtPeriodEnd"] is True
    assert saved["notificationRevision"] == 1
    assert boundaries.user.update_many.await_args.kwargs["data"] == {
        "subscriptionTier": SubscriptionTier.TRIAL
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,claimed,fingerprint",
    [
        (TrialRejectionReason.INTRO_OFFER_ALREADY_USED, False, "fp_test"),
        (TrialRejectionReason.CARD_VERIFICATION_FAILED, True, None),
    ],
)
async def test_rejected_cancel_pending_trial_is_still_canceled_immediately(
    trial, subscription, boundaries, reason, claimed, fingerprint
):
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


@pytest.mark.asyncio
async def test_scheduled_cancellation_ends_access_when_stripe_ends_the_trial(
    trial, subscription, boundaries
):
    """customer.subscription.deleted at trial end: no tier, no conversion."""
    trial = _recorded_cancel_pending(trial)
    ended_at = int(datetime.now(UTC).timestamp()) - 30
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


@pytest.mark.asyncio
async def test_a_cancel_pending_trial_ends_once_another_plan_is_live(
    trial, subscription, boundaries
):
    """A plan bought while cancel-pending ends the trial in its own cleanup,
    which can fail; the trial must never write TRIAL over that paid plan."""
    trial = _recorded_cancel_pending(trial)
    subscription["cancel_at_period_end"] = True
    ended_at = int(datetime.now(UTC).timestamp())
    canceled = {**subscription, "status": "canceled", "ended_at": ended_at}
    max_plan = stripe.ListObject.construct_from(
        {
            "data": [{"id": "sub_max", "object": "subscription", "status": "active"}],
            "has_more": False,
        },
        "test-key",
    )
    with (
        patch.object(
            fulfillment.stripe.Subscription,
            "list_async",
            AsyncMock(return_value=max_plan),
        ),
        patch.object(
            fulfillment.stripe.Subscription,
            "cancel_async",
            AsyncMock(return_value=canceled),
        ) as cancel,
    ):
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] is None
    cancel.assert_awaited_once()
    assert cancel.await_args.args == ("sub_1",)
    assert cancel.await_args.kwargs["invoice_now"] is False
    assert cancel.await_args.kwargs["prorate"] is False
    assert "cancellation_details" not in cancel.await_args.kwargs
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["status"] == "canceled"
    assert saved["rejectionReason"] is None
    boundaries.user.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "past_due"])
async def test_a_plan_that_grants_no_access_never_ends_a_cancel_pending_trial(
    trial, subscription, boundaries, status
):
    """A Checkout whose first payment failed or is still pending leaves its
    subscription incomplete for up to a day. It grants nothing, so the trial
    keeps the access it was promised until trial_end."""
    trial = _recorded_cancel_pending(trial)
    subscription["cancel_at_period_end"] = True
    unpaid_plan = stripe.ListObject.construct_from(
        {
            "data": [{"id": "sub_max", "object": "subscription", "status": status}],
            "has_more": False,
        },
        "test-key",
    )
    with (
        patch.object(
            fulfillment.stripe.Subscription,
            "list_async",
            AsyncMock(return_value=unpaid_plan),
        ),
        patch.object(
            fulfillment.stripe.Subscription, "cancel_async", AsyncMock()
        ) as cancel,
    ):
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_cancel_pending_trial_on_changed_items_ends_now(
    trial, subscription, boundaries
):
    """The reconcile refuses an unconverted trial on any price but the accepted
    one, so a cancel-pending trial whose items changed ends now, as every
    cancel-pending trial used to, instead of failing every sync."""
    subscription.update(cancel_at_period_end=True, items=None)
    canceled = {**subscription, "status": "canceled", "cancel_at_period_end": False}
    with patch.object(
        fulfillment.stripe.Subscription,
        "cancel_async",
        AsyncMock(return_value=canceled),
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_trial_not_cancel_pending_never_looks_for_other_plans(
    trial, boundaries
):
    with patch.object(
        fulfillment.stripe.Subscription, "list_async", AsyncMock()
    ) as others:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    others.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_cancel_pending_trial_keeps_access_after_its_card_expires(
    trial, subscription, boundaries
):
    """No charge is coming, so the card verified at the start is enough until
    the trial ends."""
    trial = _recorded_cancel_pending(trial)
    subscription["cancel_at_period_end"] = True
    subscription["default_payment_method"]["card"].update(exp_month=1, exp_year=2020)
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["cardVerifiedAt"] == trial.card_verified_at


@pytest.mark.parametrize(
    "pending,verified,expected",
    [
        (True, True, SubscriptionTier.TRIAL),
        (False, True, SubscriptionTier.NO_TIER),
        (True, False, SubscriptionTier.NO_TIER),
    ],
    ids=["cancel-pending", "converting", "never-verified"],
)
def test_only_a_cancel_pending_trial_skips_the_card_check(
    trial, pending, verified, expected
):
    now = datetime.now(UTC)
    trial = trial.model_copy(update={"card_verified_at": now if verified else None})
    snapshot = fulfillment.SubscriptionSnapshot(
        id="sub_1",
        customer="cus_1",
        status="trialing",
        trial_end=int(now.timestamp()) + 86400,
        cancel_at_period_end=pending,
    )
    assert fulfillment.trial_subscription_tier(trial, snapshot, now) == expected
