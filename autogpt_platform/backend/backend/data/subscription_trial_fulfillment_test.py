from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_replayed_cancellation_does_not_advance_notice_revision(
    trial, boundaries
):
    now = datetime.now(UTC)
    snapshot = fulfillment.SubscriptionSnapshot(
        id="sub_1", customer="cus_1", status="trialing", cancel_at_period_end=True
    )
    await fulfillment._save_snapshot(
        trial, snapshot, SubscriptionTier.TRIAL, now, boundaries, True
    )
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"][
            "notificationRevision"
        ]
        == 1
    )
    canceled = trial.model_copy(
        update={"cancel_at_period_end": True, "notification_revision": 1}
    )
    await fulfillment._save_snapshot(
        canceled, snapshot, SubscriptionTier.TRIAL, now, boundaries, True
    )
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"][
            "notificationRevision"
        ]
        == 1
    )


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


@pytest.mark.asyncio
async def test_reused_identity_cancels_trial_before_granting_access(
    trial, subscription, boundaries
):
    canceled = {**subscription, "status": "canceled"}
    with (
        patch.object(
            fulfillment,
            "claim_trial_identities",
            AsyncMock(return_value=False),
            create=True,
        ),
        patch.object(
            fulfillment.stripe.Subscription,
            "cancel_async",
            AsyncMock(return_value=canceled),
        ) as cancel,
    ):
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    cancel.assert_awaited_once()
    assert cancel.await_args.kwargs["invoice_now"] is False
    assert cancel.await_args.kwargs["prorate"] is False
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["status"] == "canceled"
    assert saved["consumedAt"] is not None


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
@pytest.mark.parametrize("changed_items", [False, True])
async def test_scheduled_cancellation_ends_trial_immediately_with_the_flag_off(
    trial, subscription, boundaries, changed_items
):
    subscription["cancel_at_period_end"] = True
    if changed_items:
        subscription["items"] = None
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


@pytest.mark.asyncio
async def test_canceled_subscription_revokes_access_even_if_items_changed(
    trial, subscription, boundaries
):
    subscription.update(status="canceled", items=None)
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"]["cardVerifiedAt"]
        is None
    )
