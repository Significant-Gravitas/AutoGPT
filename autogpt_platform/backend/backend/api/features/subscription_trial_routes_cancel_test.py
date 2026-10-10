"""Canceling a trial with the trial-cancel flag on schedules its end: access
holds until the trial would have ended, and the cancellation can be resumed."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features import subscription_trial_routes as routes
from backend.api.features import subscription_trial_routes_test as routes_test
from backend.data import subscription_trial_cancel as trial_cancel
from backend.data.subscription_trial import TrialState
from backend.util.feature_flag import Flag

ENDED = "This trial has ended. Manage the plan in billing."

billing_return_origin = routes_test.billing_return_origin
track_checkout_started = routes_test.track_checkout_started
cancel_flag = routes_test.cancel_flag
keeps_access_copy = routes_test.keeps_access_copy
trial = routes_test.trial


def _started(trial: TrialState, **changes) -> TrialState:
    now = datetime.now(UTC)
    return trial.model_copy(
        update={
            "subscription_id": "sub_1",
            "status": "trialing",
            "consumed_at": now,
            "card_verified_at": now,
            "started_at": now,
            "ends_at": now + timedelta(days=5),
            **changes,
        }
    )


def _live(trial: TrialState, **changes):
    return routes.stripe.Subscription.construct_from(
        {
            "id": "sub_1",
            "customer": trial.customer_id,
            "status": "trialing",
            "cancel_at_period_end": False,
            "trial_end": int((datetime.now(UTC) + timedelta(days=5)).timestamp()),
            "metadata": {"trial_enrollment_id": trial.id, "user_id": trial.user_id},
            **changes,
        },
        "test-key",
    )


@pytest.fixture
def live_stripe():
    """Stripe (a customer with no other subscription), the syncs, the checkout
    lock and the status lookups a cancel or resume reaches.
    ``lock.side_effect`` makes the checkout lock busy."""
    calls = MagicMock()
    no_others = routes.stripe.ListObject.construct_from(
        {"data": [], "has_more": False}, "test-key"
    )

    @asynccontextmanager
    async def lock(user_id: str):
        calls.lock(user_id)
        yield

    with (
        patch.object(trial_cancel, "subscription_checkout_lock", lock),
        patch.object(routes.stripe.Subscription, "retrieve_async", AsyncMock()) as get,
        patch.object(
            routes.stripe.Subscription,
            "list_async",
            AsyncMock(return_value=no_others),
        ) as others,
        patch.object(routes.stripe.Subscription, "modify_async", AsyncMock()) as modify,
        patch.object(routes.stripe.Subscription, "cancel_async", AsyncMock()) as end,
        patch.object(
            trial_cancel, "expire_other_subscription_checkouts", AsyncMock()
        ) as expire,
        patch.object(
            trial_cancel, "sync_subscription_from_stripe", AsyncMock()
        ) as sync,
        patch.object(routes, "sync_subscription_from_stripe", AsyncMock()) as old_sync,
        patch.object(
            routes, "has_received_onboarding_credit", AsyncMock(return_value=False)
        ),
    ):
        for name, mock in (
            ("retrieve", get),
            ("others", others),
            ("modify", modify),
            ("end_now", end),
            ("expire", expire),
            ("sync", sync),
            ("old_sync", old_sync),
        ):
            calls.attach_mock(mock, name)
        yield calls


@pytest.mark.asyncio
async def test_cancel_with_the_flag_on_keeps_access_until_trial_end(
    trial, cancel_flag, keeps_access_copy, live_stripe
):
    cancel_flag.return_value = (True, True)
    keeps_access_copy.return_value = True
    started = _started(trial)
    pending = _live(started, cancel_at_period_end=True)
    live_stripe.retrieve.return_value = _live(started)
    live_stripe.modify.return_value = pending
    after = started.model_copy(update={"cancel_at_period_end": True})
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(side_effect=[started, after])
    ):
        status = await routes.cancel_trial(started.user_id)
    live_stripe.modify.assert_awaited_once_with("sub_1", cancel_at_period_end=True)
    live_stripe.end_now.assert_not_awaited()
    live_stripe.sync.assert_awaited_once_with(dict(pending))
    cancel_flag.assert_awaited_once_with(
        Flag.TRIAL_CANCEL_AT_PERIOD_END, started.user_id, default=False
    )
    assert status.active and status.status == "trialing"
    assert status.cancel_at_period_end and status.cancel_keeps_access


@pytest.mark.asyncio
async def test_flag_on_cancel_after_the_trial_ended_reconciles_then_conflicts(
    trial, cancel_flag, live_stripe
):
    cancel_flag.return_value = (True, True)
    started = _started(trial)
    ended = _live(started, status="canceled")
    live_stripe.retrieve.return_value = ended
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=started)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.cancel_trial(started.user_id)
    assert (error.value.status_code, error.value.detail) == (409, ENDED)
    live_stripe.sync.assert_awaited_once_with(dict(ended))
    live_stripe.modify.assert_not_awaited()
    live_stripe.end_now.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"customer": "cus_other"},
        {"metadata": {"trial_enrollment_id": "trial-x", "user_id": "user-1"}},
    ],
    ids=["other-customer", "other-enrollment"],
)
async def test_flag_on_cancel_refuses_a_subscription_the_trial_does_not_own(
    trial, cancel_flag, live_stripe, change
):
    cancel_flag.return_value = (True, True)
    started = _started(trial)
    live_stripe.retrieve.return_value = _live(started, **change)
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=started)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.cancel_trial(started.user_id)
    assert (error.value.status_code, error.value.detail) == (409, ENDED)
    live_stripe.modify.assert_not_awaited()
    live_stripe.end_now.assert_not_awaited()
    live_stripe.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_on_cancel_stripe_failure_is_retryable_without_a_sync(
    trial, cancel_flag, live_stripe
):
    cancel_flag.return_value = (True, True)
    started = _started(trial)
    live_stripe.retrieve.return_value = _live(started)
    live_stripe.modify.side_effect = routes.stripe.APIConnectionError("unreachable")
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=started)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.cancel_trial(started.user_id)
    assert error.value.status_code == 502
    assert error.value.detail == "Unable to cancel your trial. Please retry."
    live_stripe.sync.assert_not_awaited()
    live_stripe.old_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_on_cancel_of_a_converted_trial_is_conflict(
    trial, cancel_flag, live_stripe
):
    cancel_flag.return_value = (True, True)
    converted = _started(trial, converted_at=datetime.now(UTC))
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=converted)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.cancel_trial(converted.user_id)
    assert error.value.status_code == 409
    live_stripe.retrieve.assert_not_awaited()
    cancel_flag.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [(False, False), (True, False)])
async def test_an_unreadable_flag_refuses_to_cancel_before_touching_stripe(
    trial, cancel_flag, live_stripe, value
):
    """Guessing "off" would end, for good, a trial the person was promised."""
    cancel_flag.return_value = value
    started = _started(trial)
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=started)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.cancel_trial(started.user_id)
    assert error.value.status_code == 502
    assert error.value.detail == "Unable to cancel your trial. Please retry."
    assert live_stripe.mock_calls == []
