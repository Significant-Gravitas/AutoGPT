"""Billing lifecycle emails for a trial's own subscription: converting a
cancel-pending trial early is not a paid resume, while a converted trial that
is canceled later is a paid cancellation."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.notifications import lifecycle, lifecycle_test, trial_test
from backend.notifications.lifecycle_test import _run, _subscription, _User

trial = trial_test.trial
fields_on = lifecycle_test.fields_on


def _trial_subscription(**over) -> dict:
    return _subscription(
        status="active",
        metadata={"trial_enrollment_id": "trial-1", "user_id": "user-1"},
        **over,
    )


async def _run_converted(trial, coro_factory):
    converted = trial.model_copy(update={"converted_at": datetime.now(timezone.utc)})
    trials = MagicMock(get_subscription_trial=AsyncMock(return_value=converted))
    with patch("backend.notifications.trial.credit_db", return_value=trials):
        return await _run(coro_factory, _User())


@pytest.mark.asyncio
async def test_converting_a_cancel_pending_trial_early_is_not_a_paid_resume(
    trial, fields_on
):
    """Subscribe now on a cancel-pending trial ends the trial and clears its
    cancellation in one update. The trial's conversion notice covers it."""
    calls = await _run_converted(
        trial,
        lambda: lifecycle.on_subscription_updated(
            _trial_subscription(),
            {"cancel_at_period_end": True, "status": "trialing"},
        ),
    )
    calls["claim"].assert_not_awaited()
    calls["notify"].assert_not_awaited()
    fields_on.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_converted_trial_canceled_later_is_a_paid_cancellation(trial):
    calls = await _run_converted(
        trial,
        lambda: lifecycle.on_subscription_updated(
            _trial_subscription(cancel_at_period_end=True, canceled_at=1789100000),
            {"cancel_at_period_end": False},
        ),
    )
    queued = calls["notify"].await_args.args[0]
    assert queued.type is NotificationType.SUBSCRIPTION_CANCELLED
