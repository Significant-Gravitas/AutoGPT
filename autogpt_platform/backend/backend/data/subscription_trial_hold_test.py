from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_checkout as checkout
from backend.data import subscription_trial_eligibility as eligibility
from backend.data import subscription_trial_stripe as fulfillment
from backend.data.subscription_checkout import SubscriptionCheckoutUnavailable
from backend.data.subscription_trial_rejection import TrialRejectionReason

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.mark.asyncio
async def test_authorized_hold_is_released_and_the_trial_starts(
    trial, boundaries, card_hold
):
    with patch.object(stripe.Subscription, "cancel_async", AsyncMock()) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    hold = card_hold.create.await_args.kwargs
    assert (hold["amount"], hold["currency"]) == (100, "usd")
    assert (hold["customer"], hold["payment_method"]) == ("cus_1", "pm_1")
    assert hold["capture_method"] == "manual"
    assert hold["confirm"] is True and hold["off_session"] is True
    assert card_hold.cancel.await_args.args == ("pi_hold",)
    cancel.assert_not_awaited()
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["consumedAt"] is not None and saved["rejectionReason"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,decline_code",
    [
        ("card_declined", "insufficient_funds"),
        ("card_declined", "transaction_not_allowed"),
        ("expired_card", None),
        ("incorrect_number", None),
    ],
)
async def test_declined_hold_cancels_the_trial_and_leaves_the_offer_unused(
    trial, subscription, boundaries, card_hold, code, decline_code
):
    card_hold.create.side_effect = card_error(code, decline_code)
    canceled = {**subscription, "status": "canceled"}
    with patch.object(
        stripe.Subscription, "cancel_async", AsyncMock(return_value=canceled)
    ) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    assert cancel.await_args.kwargs["invoice_now"] is False
    assert cancel.await_args.kwargs["prorate"] is False
    assert cancel.await_args.kwargs["cancellation_details"] == {
        "comment": "autogpt_trial:card_declined"
    }
    card_hold.cancel.assert_not_awaited()
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["status"] == "checkout_pending"
    assert saved["rejectionReason"] == TrialRejectionReason.CARD_DECLINED
    assert saved["checkoutAttempt"] == {"increment": 1}
    assert saved["stripeCheckoutSessionId"] is None
    assert saved.get("consumedAt") is None
    tier = boundaries.user.update_many.await_args.kwargs["data"]["subscriptionTier"]
    assert tier == SubscriptionTier.NO_TIER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,decline_code",
    [
        ("authentication_required", "authentication_required"),
        ("card_declined", "authentication_required"),
    ],
)
async def test_hold_needing_authentication_still_starts_the_trial(
    trial, boundaries, card_hold, code, decline_code
):
    card_hold.create.side_effect = card_error(code, decline_code)
    with patch.object(stripe.Subscription, "cancel_async", AsyncMock()) as cancel:
        result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()
    card_hold.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_replayed_webhook_after_the_trial_started_places_no_second_hold(
    trial, boundaries, card_hold
):
    now = datetime.now(UTC)
    started = trial.model_copy(
        update={
            "status": "trialing",
            "subscription_id": "sub_1",
            "card_verified_at": now,
            "consumed_at": now,
        }
    )
    result = await fulfillment._reconcile_locked(started, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    card_hold.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_hold_retried_after_a_failed_release_reuses_its_idempotency_keys(
    trial, boundaries, card_hold
):
    card_hold.cancel.side_effect = [
        stripe.APIConnectionError("Stripe is unreachable"),
        card_hold.cancel.return_value,
    ]
    with pytest.raises(stripe.APIConnectionError):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    first, second = card_hold.create.await_args_list
    assert first.kwargs["idempotency_key"] == second.kwargs["idempotency_key"]
    released = [
        call.kwargs["idempotency_key"] for call in card_hold.cancel.await_args_list
    ]
    assert released[0] == released[1]


@pytest.mark.asyncio
async def test_next_attempt_after_a_declined_hold_is_judged_afresh(
    trial, subscription, session, boundaries, card_hold
):
    retry = trial.model_copy(
        update={
            "checkout_attempt": 1,
            "rejection_reason": TrialRejectionReason.CARD_DECLINED,
        }
    )
    subscription["metadata"]["trial_checkout_attempt"] = "1"
    session["metadata"]["trial_checkout_attempt"] = "1"
    with patch.object(stripe.Subscription, "cancel_async", AsyncMock()) as cancel:
        result = await fulfillment._reconcile_locked(retry, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    cancel.assert_not_awaited()
    card_hold.create.assert_awaited_once()
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["rejectionReason"] is None and saved["consumedAt"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt,eligible", [(1, True), (0, False)])
async def test_only_a_reopened_checkout_forgives_its_declined_subscription(
    trial, session, attempt, eligible
):
    current = trial.model_copy(
        update={"checkout_attempt": attempt, "checkout_session_id": None}
    )
    declined = {"id": "sub_1", "status": "canceled"}
    user = SimpleNamespace(
        created_at=datetime.now(UTC), subscription_tier=SubscriptionTier.NO_TIER
    )
    with (
        patch.object(eligibility, "get_user_by_id", AsyncMock(return_value=user)),
        patch.object(
            stripe.Subscription, "list_async", AsyncMock(return_value=page(declined))
        ),
        patch.object(
            stripe.checkout.Session, "list_async", AsyncMock(return_value=page(session))
        ),
    ):
        verification = eligibility.verify_trial_eligibility(
            "user-1", "cus_1", trial.offer, trial=current
        )
        if eligible:
            await verification
        else:
            with pytest.raises(SubscriptionCheckoutUnavailable):
                await verification


@pytest.mark.asyncio
async def test_returning_from_checkout_after_a_declined_hold_reports_the_status(
    trial,
):
    reopened = trial.model_copy(
        update={
            "checkout_attempt": 1,
            "checkout_session_id": None,
            "rejection_reason": TrialRejectionReason.CARD_DECLINED,
        }
    )
    with (
        patch.object(
            checkout, "get_subscription_trial", AsyncMock(return_value=reopened)
        ),
        patch.object(
            stripe.checkout.Session, "list_async", AsyncMock(return_value=page())
        ),
    ):
        await checkout.confirm_trial_checkout("user-1")


def card_error(code: str, decline_code: str | None) -> stripe.CardError:
    return stripe.CardError(
        "Your card was declined.",
        None,
        code,
        json_body={
            "error": {"type": "card_error", "code": code, "decline_code": decline_code}
        },
    )


def page(*items: dict) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {"data": list(items), "has_more": False}, "test-key"
    )
