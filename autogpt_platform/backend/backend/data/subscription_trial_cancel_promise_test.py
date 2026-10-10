"""A cancel the person confirmed under "you keep full access" records that
promise on the trial subscription, so no reconcile ends the trial early."""

import pytest
import stripe

from backend.data import subscription_trial_cancel as cancel
from backend.data import subscription_trial_cancel_test as cancel_test
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_cancel_test import _assert_no_writes, _live

PROMISED = {"trial_cancel_keeps_access": "true"}

trial = cancel_test.trial
api = cancel_test.api


def _scheduled(trial: TrialState, *, promised: bool) -> stripe.Subscription:
    subscription = _live(trial, cancel_at_period_end=True)
    if promised:
        subscription["metadata"].update(PROMISED)
    return subscription


@pytest.mark.asyncio
async def test_cancel_promising_access_records_the_promise_on_the_trial(trial, api):
    """Reconcile reads the promise there, so a flag turned off since the person
    confirmed never ends this trial early."""
    pending = _scheduled(trial, promised=True)
    api.retrieve.return_value = _live(trial)
    api.modify.return_value = pending
    await cancel.schedule_trial_cancellation(trial, keeps_access=True)
    api.modify.assert_awaited_once_with(
        "sub_1", cancel_at_period_end=True, metadata=PROMISED
    )
    api.sync.assert_awaited_once_with(dict(pending))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keeps_access,promised",
    [(False, False), (False, True), (True, True)],
    ids=["no-promise", "earlier-promise", "promise-already-recorded"],
)
async def test_a_second_cancel_writes_nothing_to_stripe(
    trial, api, keeps_access, promised
):
    pending = _scheduled(trial, promised=promised)
    api.retrieve.return_value = pending
    await cancel.schedule_trial_cancellation(trial, keeps_access=keeps_access)
    _assert_no_writes(api)
    api.sync.assert_awaited_once_with(dict(pending))


@pytest.mark.asyncio
async def test_a_promise_is_recorded_on_an_end_already_scheduled_elsewhere(trial, api):
    """An end scheduled in the billing portal carries no promise; one the
    person confirmed in the app must still hold."""
    promised = _scheduled(trial, promised=True)
    api.retrieve.return_value = _scheduled(trial, promised=False)
    api.modify.return_value = promised
    await cancel.schedule_trial_cancellation(trial, keeps_access=True)
    api.modify.assert_awaited_once_with(
        "sub_1", cancel_at_period_end=True, metadata=PROMISED
    )
    api.sync.assert_awaited_once_with(dict(promised))


@pytest.mark.asyncio
async def test_resume_takes_back_the_promise_with_the_end(trial, api):
    """The promise belonged to that cancellation: a later one makes its own."""
    resumed = _live(trial)
    api.retrieve.return_value = _scheduled(trial, promised=True)
    api.modify.return_value = resumed
    await cancel.resume_trial_subscription(trial)
    api.modify.assert_awaited_once_with(
        "sub_1",
        cancel_at_period_end=False,
        metadata={"trial_cancel_keeps_access": ""},
    )
    api.sync.assert_awaited_once_with(dict(resumed))
