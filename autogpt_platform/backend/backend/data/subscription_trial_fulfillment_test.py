from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_stripe as fulfillment

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


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
