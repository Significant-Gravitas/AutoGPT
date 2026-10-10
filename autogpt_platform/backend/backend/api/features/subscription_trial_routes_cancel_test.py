"""Canceling a trial with the trial-cancel flag on schedules its end: access
holds until the trial would have ended, and the cancellation can be resumed."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.features import subscription_trial_routes as routes
from backend.api.features import subscription_trial_routes_test as routes_test
from backend.api.features.subscription_trial_models import TrialCancelRequest
from backend.data import subscription_trial_cancel as trial_cancel
from backend.data.subscription_trial import TrialState
from backend.util.feature_flag import Flag

ENDED = "This trial has ended. Manage the plan in billing."
PROMISED = {"trial_cancel_keeps_access": "true"}
FLAG_READS = {
    "on": (True, True),
    "off": (False, True),
    "unreadable": (False, False),
    "unreadable-on": (True, False),
}

billing_return_origin = routes_test.billing_return_origin
track_checkout_started = routes_test.track_checkout_started
cancel_flag = routes_test.cancel_flag
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
    trial, cancel_flag, live_stripe
):
    cancel_flag.return_value = (True, True)
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
    cancel_flag.assert_awaited_with(
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


async def _cancel(trial: TrialState, live_stripe, keeps_access: bool | None) -> None:
    started = _started(trial)
    live_stripe.retrieve.return_value = _live(started)
    live_stripe.modify.return_value = _live(started, cancel_at_period_end=True)
    live_stripe.end_now.return_value = _live(started, status="canceled")
    body = (
        None if keeps_access is None else TrialCancelRequest(keeps_access=keeps_access)
    )
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=started)
    ):
        await routes.cancel_trial(started.user_id, body)


BODIES = {"no-body": None, "ends-now-copy": False, "keeps-access-copy": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,keeps_access",
    [
        pytest.param(FLAG_READS[read], BODIES[copy], id=f"{read}-{copy}")
        for read in FLAG_READS
        for copy in BODIES
        if read != "off" or BODIES[copy]
    ],
)
async def test_cancel_schedules_the_end_unless_nothing_says_access_is_kept(
    trial, cancel_flag, live_stripe, flag, keeps_access
):
    """A scheduled end never charges and can be taken back; ending at once
    cannot. An unreadable flag, or a person told they keep access, schedules."""
    cancel_flag.return_value = flag
    await _cancel(trial, live_stripe, keeps_access)
    promise = {"metadata": PROMISED} if keeps_access else {}
    live_stripe.modify.assert_awaited_once_with(
        "sub_1", cancel_at_period_end=True, **promise
    )
    live_stripe.end_now.assert_not_awaited()
    live_stripe.old_sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("keeps_access", [None, False], ids=["no-body", "ends-now"])
async def test_cancel_ends_the_trial_now_only_on_an_authoritative_off(
    trial, cancel_flag, live_stripe, keeps_access
):
    cancel_flag.return_value = (False, True)
    await _cancel(trial, live_stripe, keeps_access)
    live_stripe.end_now.assert_awaited_once_with(
        "sub_1", invoice_now=False, prorate=False
    )
    live_stripe.modify.assert_not_awaited()
    live_stripe.old_sync.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_body,scheduled",
    [({"json": {"keeps_access": True}}, True), ({}, False), ({"json": {}}, False)],
    ids=["keeps-access", "no-body", "empty-body"],
)
async def test_cancel_reads_the_promise_from_the_request_body(
    trial, cancel_flag, live_stripe, request_body, scheduled
):
    started = _started(trial)
    live_stripe.retrieve.return_value = _live(started)
    live_stripe.modify.return_value = _live(started, cancel_at_period_end=True)
    live_stripe.end_now.return_value = _live(started, status="canceled")
    with patch.object(
        routes, "get_subscription_trial", AsyncMock(return_value=started)
    ):
        async with AsyncClient(
            transport=ASGITransport(app=routes_test._app(started)),
            base_url="https://example.com",
        ) as client:
            response = await client.post("/credits/trial/cancel", **request_body)
    assert response.status_code == 200
    assert live_stripe.modify.await_count == int(scheduled)
    assert live_stripe.end_now.await_count == int(not scheduled)
