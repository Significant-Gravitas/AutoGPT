from unittest.mock import AsyncMock, patch

import pytest

from backend.data import subscription_trial_stripe as fulfillment

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.mark.asyncio
@pytest.mark.parametrize("comment", [None, "I want to cancel", "unrecognized_reason"])
async def test_ordinary_cancellation_is_not_labeled_as_rejection(
    trial, subscription, boundaries, comment
):
    subscription.update(status="canceled", cancellation_details={"comment": comment})
    await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"]["rejectionReason"]
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_fingerprint", [False, True])
async def test_auto_cancellation_records_rejection_reason(
    trial, subscription, boundaries, missing_fingerprint
):
    reason = (
        "card_verification_failed"
        if missing_fingerprint
        else "intro_offer_already_used"
    )
    if missing_fingerprint:
        subscription["default_payment_method"]["card"]["fingerprint"] = None
    canceled = {**subscription, "status": "canceled"}
    with (
        patch.object(
            fulfillment, "claim_trial_identities", AsyncMock(return_value=False)
        ),
        patch.object(
            fulfillment.stripe.Subscription,
            "cancel_async",
            AsyncMock(return_value=canceled),
        ) as cancel,
    ):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert cancel.await_args.kwargs["cancellation_details"] == {
        "comment": f"autogpt_trial:{reason}"
    }
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["rejectionReason"] == reason


@pytest.mark.asyncio
async def test_reconciliation_recovers_rejection_from_stripe(
    trial, subscription, boundaries
):
    subscription.update(
        status="canceled",
        cancellation_details={"comment": "autogpt_trial:intro_offer_already_used"},
    )
    await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"]["rejectionReason"]
        == "intro_offer_already_used"
    )


@pytest.mark.asyncio
async def test_reconciliation_preserves_rejection_on_replay(
    trial, subscription, boundaries
):
    trial = trial.model_copy(update={"rejection_reason": "intro_offer_already_used"})
    subscription.update(status="canceled")
    await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert (
        boundaries.subscriptiontrial.update.await_args.kwargs["data"]["rejectionReason"]
        == "intro_offer_already_used"
    )
