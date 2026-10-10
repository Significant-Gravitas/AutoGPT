from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import stripe

from backend.data import subscription_trial_cancel as cancel
from backend.data.subscription_checkout import SubscriptionCheckoutUnavailable
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer

ENDED = "This trial has ended. Manage the plan in billing."


@pytest.fixture
def trial() -> TrialState:
    now = datetime.now(UTC)
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="cancel-v1",
            new_users_from=now - timedelta(days=3),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=5000,
            currency="usd",
        ),
        checkout_session_id="cs_1",
        subscription_id="sub_1",
        checkout_attempt=0,
        success_url="https://example.com/ok",
        cancel_url="https://example.com/no",
        checkout_metadata={},
        status="trialing",
        card_verified_at=now - timedelta(days=2),
        started_at=now - timedelta(days=2),
        ends_at=now + timedelta(days=5),
        consumed_at=now - timedelta(days=2),
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


def _live(trial: TrialState, **changes) -> stripe.Subscription:
    return stripe.Subscription.construct_from(
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
def api():
    """Stripe, the sync, the other-plan check and the checkout lock as this
    module calls them, recorded in call order. ``api.lock.side_effect`` makes
    the lock busy; ``api.other_plan.return_value = True`` makes another plan
    live."""
    calls = MagicMock()

    @asynccontextmanager
    async def lock(user_id: str):
        calls.lock(user_id)
        try:
            yield
        finally:
            calls.unlock()

    with (
        patch.object(cancel, "subscription_checkout_lock", lock),
        patch.object(stripe.Subscription, "retrieve_async", AsyncMock()) as retrieve,
        patch.object(stripe.Subscription, "modify_async", AsyncMock()) as modify,
        patch.object(stripe.Subscription, "cancel_async", AsyncMock()) as end_now,
        patch.object(
            cancel, "expire_other_subscription_checkouts", AsyncMock()
        ) as expire,
        patch.object(
            cancel, "other_plan_is_live", AsyncMock(return_value=False)
        ) as other_plan,
        patch.object(cancel, "sync_subscription_from_stripe", AsyncMock()) as sync,
    ):
        for name, mock in (
            ("retrieve", retrieve),
            ("modify", modify),
            ("end_now", end_now),
            ("expire", expire),
            ("other_plan", other_plan),
            ("sync", sync),
        ):
            calls.attach_mock(mock, name)
        yield calls


def _assert_no_writes(api) -> None:
    api.modify.assert_not_awaited()
    api.end_now.assert_not_awaited()
    api.expire.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_schedules_the_end_instead_of_ending_the_trial(trial, api):
    pending = _live(trial, cancel_at_period_end=True)
    api.retrieve.return_value = _live(trial)
    api.modify.return_value = pending
    await cancel.schedule_trial_cancellation(trial)
    api.modify.assert_awaited_once_with("sub_1", cancel_at_period_end=True)
    api.end_now.assert_not_awaited()
    api.sync.assert_awaited_once_with(dict(pending))


@pytest.mark.asyncio
async def test_a_second_cancel_writes_nothing_to_stripe(trial, api):
    pending = _live(trial, cancel_at_period_end=True)
    api.retrieve.return_value = pending
    await cancel.schedule_trial_cancellation(trial)
    _assert_no_writes(api)
    api.sync.assert_awaited_once_with(dict(pending))


@pytest.mark.asyncio
async def test_cancel_holds_the_checkout_lock_from_read_to_sync(trial, api):
    """Subscribe now converts the trial under this lock: a cancel landing in
    between would schedule the new paid plan to end."""
    pending = _live(trial, cancel_at_period_end=True)
    api.retrieve.return_value = _live(trial)
    api.modify.return_value = pending
    await cancel.schedule_trial_cancellation(trial)
    assert api.mock_calls == [
        call.lock("user-1"),
        call.retrieve("sub_1"),
        call.modify("sub_1", cancel_at_period_end=True),
        call.sync(dict(pending)),
        call.unlock(),
    ]


@pytest.mark.asyncio
async def test_cancel_while_the_trial_is_being_updated_is_refused_untouched(trial, api):
    api.lock.side_effect = SubscriptionCheckoutUnavailable(
        "Another checkout is already starting. Please retry."
    )
    with pytest.raises(
        cancel.TrialChangeRefused,
        match="^Your trial is already being updated. Please retry.$",
    ):
        await cancel.schedule_trial_cancellation(trial)
    api.retrieve.assert_not_awaited()
    _assert_no_writes(api)
    api.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_takes_back_the_scheduled_end(trial, api):
    resumed = _live(trial)
    api.retrieve.return_value = _live(trial, cancel_at_period_end=True)
    api.modify.return_value = resumed
    await cancel.resume_trial_subscription(trial)
    api.modify.assert_awaited_once_with("sub_1", cancel_at_period_end=False)
    api.sync.assert_awaited_once_with(dict(resumed))


@pytest.mark.asyncio
async def test_resume_expires_other_open_checkouts_before_resuming(trial, api):
    """A plan checkout opened while cancel-pending must not complete beside the
    resumed trial and leave two live subscriptions."""
    api.retrieve.return_value = _live(trial, cancel_at_period_end=True)
    api.modify.return_value = _live(trial)
    await cancel.resume_trial_subscription(trial)
    writes = [c for c in api.mock_calls if c[0] in ("expire", "modify")]
    assert writes == [
        call.expire("cus_1"),
        call.modify("sub_1", cancel_at_period_end=False),
    ]


@pytest.mark.asyncio
async def test_resume_holds_the_checkout_lock_from_read_to_sync(trial, api):
    """No plan checkout can open between the expiry and the resume, where it
    would escape the expiry and could complete beside the resumed trial."""
    resumed = _live(trial)
    api.retrieve.return_value = _live(trial, cancel_at_period_end=True)
    api.modify.return_value = resumed
    await cancel.resume_trial_subscription(trial)
    assert api.mock_calls == [
        call.lock("user-1"),
        call.retrieve("sub_1"),
        call.expire("cus_1"),
        call.other_plan("cus_1", "sub_1"),
        call.modify("sub_1", cancel_at_period_end=False),
        call.sync(dict(resumed)),
        call.unlock(),
    ]


@pytest.mark.asyncio
async def test_resume_is_refused_while_another_plan_is_live(trial, api):
    """A plan bought through Checkout while cancel-pending ends the trial only
    once its webhook or the stale-subscription cleanup runs; resuming before
    then would let the trial convert next to it and bill twice."""
    api.retrieve.return_value = _live(trial, cancel_at_period_end=True)
    api.other_plan.return_value = True
    with pytest.raises(
        cancel.TrialChangeRefused,
        match="^Another plan is already active. Manage it in billing.$",
    ):
        await cancel.resume_trial_subscription(trial)
    assert api.mock_calls == [
        call.lock("user-1"),
        call.retrieve("sub_1"),
        call.expire("cus_1"),
        call.other_plan("cus_1", "sub_1"),
        call.unlock(),
    ]


@pytest.mark.asyncio
async def test_resume_while_a_checkout_is_starting_is_refused_untouched(trial, api):
    """Another tab resuming, or a plan change starting, holds the lock."""
    api.lock.side_effect = SubscriptionCheckoutUnavailable(
        "Another checkout is already starting. Please retry."
    )
    with pytest.raises(
        cancel.TrialChangeRefused,
        match="^Your trial is already being updated. Please retry.$",
    ):
        await cancel.resume_trial_subscription(trial)
    api.retrieve.assert_not_awaited()
    _assert_no_writes(api)
    api.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_without_a_scheduled_end_reconciles_then_writes_nothing(
    trial, api
):
    """The saved trial can still say cancel-pending (a renewal in the billing
    portal whose webhook has not landed): the sync lets the reload show it."""
    live = _live(trial)
    api.retrieve.return_value = live
    with pytest.raises(cancel.TrialChangeRefused, match="^Nothing to resume.$"):
        await cancel.resume_trial_subscription(trial)
    _assert_no_writes(api)
    api.sync.assert_awaited_once_with(dict(live))


@pytest.mark.asyncio
async def test_resume_after_trial_end_is_reconciled_then_refused(trial, api):
    past = int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())
    live = _live(trial, cancel_at_period_end=True, trial_end=past)
    api.retrieve.return_value = live
    with pytest.raises(cancel.TrialChangeRefused) as refused:
        await cancel.resume_trial_subscription(trial)
    assert str(refused.value) == ENDED
    _assert_no_writes(api)
    api.sync.assert_awaited_once_with(dict(live))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        lambda trial: trial.model_copy(update={"converted_at": datetime.now(UTC)}),
        lambda trial: trial.model_copy(update={"consumed_at": None}),
        lambda trial: trial.model_copy(update={"subscription_id": None}),
        lambda trial: None,
    ],
    ids=["converted", "not-started", "no-subscription", "no-trial"],
)
async def test_resume_needs_a_started_unconverted_trial(trial, api, change):
    with pytest.raises(cancel.TrialChangeRefused, match="^Nothing to resume.$"):
        await cancel.resume_trial_subscription(change(trial))
    api.retrieve.assert_not_awaited()
    _assert_no_writes(api)


OPERATIONS = [
    pytest.param(cancel.schedule_trial_cancellation, False, id="cancel"),
    pytest.param(cancel.resume_trial_subscription, True, id="resume"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,pending", OPERATIONS)
@pytest.mark.parametrize("status", ["canceled", "active", "past_due"])
async def test_a_trial_stripe_no_longer_runs_is_reconciled_then_refused(
    trial, api, operation, pending, status
):
    """Stripe ended or converted it before its webhook landed: the sync lets
    the status the person reloads show what happened."""
    ended = _live(trial, status=status, cancel_at_period_end=pending)
    api.retrieve.return_value = ended
    with pytest.raises(cancel.TrialChangeRefused) as refused:
        await operation(trial)
    assert str(refused.value) == ENDED
    api.sync.assert_awaited_once_with(dict(ended))
    _assert_no_writes(api)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,pending", OPERATIONS)
async def test_a_trial_ending_mid_update_is_reconciled_then_refused(
    trial, api, operation, pending
):
    ended = _live(trial, status="canceled", cancel_at_period_end=pending)
    api.retrieve.side_effect = [_live(trial, cancel_at_period_end=pending), ended]
    api.modify.side_effect = stripe.InvalidRequestError(
        "A canceled subscription can only update its cancellation_details", None
    )
    with pytest.raises(cancel.TrialChangeRefused) as refused:
        await operation(trial)
    assert str(refused.value) == ENDED
    api.sync.assert_awaited_once_with(dict(ended))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,pending", OPERATIONS)
async def test_a_rejected_update_on_a_live_trial_stays_retryable(
    trial, api, operation, pending
):
    api.retrieve.return_value = _live(trial, cancel_at_period_end=pending)
    api.modify.side_effect = stripe.InvalidRequestError("Try again", None)
    with pytest.raises(stripe.InvalidRequestError):
        await operation(trial)
    api.sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,pending", OPERATIONS)
async def test_a_stripe_failure_changes_nothing(trial, api, operation, pending):
    api.retrieve.return_value = _live(trial, cancel_at_period_end=pending)
    api.modify.side_effect = stripe.APIConnectionError("Stripe is unreachable")
    with pytest.raises(stripe.StripeError):
        await operation(trial)
    api.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_stops_if_open_checkouts_cannot_be_expired(trial, api):
    api.retrieve.return_value = _live(trial, cancel_at_period_end=True)
    api.expire.side_effect = stripe.APIConnectionError("Stripe is unreachable")
    with pytest.raises(stripe.StripeError):
        await cancel.resume_trial_subscription(trial)
    api.modify.assert_not_awaited()
    api.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_stops_if_other_plans_cannot_be_listed(trial, api):
    api.retrieve.return_value = _live(trial, cancel_at_period_end=True)
    api.other_plan.side_effect = stripe.APIConnectionError("Stripe is unreachable")
    with pytest.raises(stripe.StripeError):
        await cancel.resume_trial_subscription(trial)
    api.modify.assert_not_awaited()
    api.sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,pending", OPERATIONS)
@pytest.mark.parametrize(
    "change",
    [
        {"customer": "cus_other"},
        {"metadata": {"trial_enrollment_id": "trial-x", "user_id": "user-1"}},
        {"metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-x"}},
        {"metadata": None},
    ],
    ids=["other-customer", "other-enrollment", "other-user", "no-metadata"],
)
async def test_only_this_enrollments_live_trial_can_change(
    trial, api, operation, pending, change
):
    api.retrieve.return_value = _live(trial, cancel_at_period_end=pending, **change)
    with pytest.raises(cancel.TrialChangeRefused) as refused:
        await operation(trial)
    assert str(refused.value) == ENDED
    _assert_no_writes(api)
    api.sync.assert_not_awaited()
