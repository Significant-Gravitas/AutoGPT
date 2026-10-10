"""Resuming a scheduled trial cancellation, and the status copy that says
whether canceling keeps access until the trial ends."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.api.features import subscription_trial_routes as routes
from backend.api.features import subscription_trial_routes_cancel_test as cancel_test
from backend.api.features import subscription_trial_routes_test as routes_test
from backend.api.features.subscription_trial_routes_cancel_test import (
    ENDED,
    _live,
    _started,
)
from backend.data.subscription_checkout import SubscriptionCheckoutUnavailable
from backend.util.feature_flag import Flag

billing_return_origin = routes_test.billing_return_origin
track_checkout_started = routes_test.track_checkout_started
cancel_flag = routes_test.cancel_flag
keeps_access_copy = routes_test.keeps_access_copy
trial = routes_test.trial
live_stripe = cancel_test.live_stripe


@pytest.mark.asyncio
async def test_resume_takes_back_a_scheduled_cancellation(
    trial, cancel_flag, live_stripe
):
    """Resume follows the trial's own state, never the flag: a trial canceled
    while the flag was on stays resumable after it is turned off."""
    pending = _started(trial, cancel_at_period_end=True)
    resumed = _live(pending)
    live_stripe.retrieve.return_value = _live(pending, cancel_at_period_end=True)
    live_stripe.modify.return_value = resumed
    after = pending.model_copy(update={"cancel_at_period_end": False})
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(side_effect=[pending, after])
    ):
        status = await routes.resume_trial(pending.user_id)
    live_stripe.expire.assert_awaited_once_with("cus_1")
    live_stripe.modify.assert_awaited_once_with("sub_1", cancel_at_period_end=False)
    live_stripe.sync.assert_awaited_once_with(dict(resumed))
    cancel_flag.assert_not_awaited()
    assert status.active and not status.cancel_at_period_end


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,live,detail",
    [
        ({}, {}, "Nothing to resume."),
        ({"converted_at": datetime.now(UTC)}, None, "Nothing to resume."),
        ({}, {"status": "canceled", "cancel_at_period_end": True}, ENDED),
    ],
    ids=["not-pending", "converted", "ended"],
)
async def test_resume_without_a_live_scheduled_cancellation_is_conflict(
    trial, live_stripe, changes, live, detail
):
    current = _started(trial, cancel_at_period_end=True, **changes)
    live_stripe.retrieve.return_value = None if live is None else _live(current, **live)
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=current)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.resume_trial(current.user_id)
    assert (error.value.status_code, error.value.detail) == (409, detail)
    live_stripe.modify.assert_not_awaited()
    live_stripe.expire.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_while_a_checkout_is_starting_is_a_retryable_conflict(
    trial, live_stripe
):
    pending = _started(trial, cancel_at_period_end=True)
    live_stripe.lock.side_effect = SubscriptionCheckoutUnavailable(
        "Another checkout is already starting. Please retry."
    )
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=pending)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.resume_trial(pending.user_id)
    assert (error.value.status_code, error.value.detail) == (
        409,
        "Another checkout is already starting. Please retry.",
    )
    live_stripe.retrieve.assert_not_awaited()
    live_stripe.expire.assert_not_awaited()
    live_stripe.modify.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_with_another_plan_live_is_conflict_untouched(trial, live_stripe):
    """A plan bought while the trial was cancel-pending must not be joined by
    the trial converting at its end."""
    pending = _started(trial, cancel_at_period_end=True)
    live_stripe.retrieve.return_value = _live(pending, cancel_at_period_end=True)
    live_stripe.others.return_value = routes.stripe.ListObject.construct_from(
        {
            "data": [{"id": "sub_max", "object": "subscription", "status": "active"}],
            "has_more": False,
        },
        "test-key",
    )
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=pending)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.resume_trial(pending.user_id)
    assert (error.value.status_code, error.value.detail) == (
        409,
        "Another plan is already active. Manage it in billing.",
    )
    live_stripe.modify.assert_not_awaited()
    live_stripe.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_stripe_failure_is_retryable_without_a_sync(trial, live_stripe):
    pending = _started(trial, cancel_at_period_end=True)
    live_stripe.retrieve.return_value = _live(pending, cancel_at_period_end=True)
    live_stripe.modify.side_effect = routes.stripe.APIConnectionError("unreachable")
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=pending)
    ):
        with pytest.raises(routes.HTTPException) as error:
            await routes.resume_trial(pending.user_id)
    assert error.value.status_code == 502
    assert error.value.detail == "Unable to resume your trial. Please retry."
    live_stripe.sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_trial_status_says_whether_canceling_keeps_access(
    trial, keeps_access_copy, enabled
):
    keeps_access_copy.return_value = enabled
    with (
        patch.object(
            routes, "get_subscription_trial", AsyncMock(return_value=_started(trial))
        ),
        patch.object(
            routes, "has_received_onboarding_credit", AsyncMock(return_value=False)
        ),
    ):
        status = await routes.get_trial_status(trial.user_id)
    assert status.cancel_keeps_access is enabled
    keeps_access_copy.assert_awaited_once_with(
        Flag.TRIAL_CANCEL_AT_PERIOD_END, trial.user_id, default=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_trial_offer_says_whether_canceling_keeps_access(
    trial, keeps_access_copy, enabled
):
    keeps_access_copy.return_value = enabled
    user = MagicMock(
        stripe_customer_id=None,
        created_at=datetime.now(UTC),
        subscription_tier=SubscriptionTier.NO_TIER,
    )
    with (
        patch.object(routes, "get_subscription_trial", AsyncMock(return_value=None)),
        patch.object(routes, "get_trial_offer", AsyncMock(return_value=trial.offer)),
        patch.object(routes, "trial_seat_available", AsyncMock(return_value=True)),
        patch.object(routes, "get_user_by_id", AsyncMock(return_value=user)),
        patch.object(
            routes, "resolve_trial_price", AsyncMock(return_value=trial.offer)
        ),
        patch.object(
            routes, "has_received_onboarding_credit", AsyncMock(return_value=False)
        ),
    ):
        status = await routes.get_trial_status(trial.user_id)
    assert status.eligible
    assert status.cancel_keeps_access is enabled
