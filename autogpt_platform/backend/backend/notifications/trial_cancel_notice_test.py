"""Trial notices around a scheduled cancellation: the canceled email keeps
access until the trial ends, and each cancel or resume flip sends the notice
for the trial's current state, never for one that has already converted."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.subscription_trial import TrialState
from backend.notifications import trial as notices
from backend.notifications import trial_test
from backend.notifications.renderer import render

trial = trial_test.trial
urls = trial_test.urls


def test_canceled_notice_keeps_access_until_the_trial_ends(trial, urls):
    data = notices.trial_notice_data(trial, "canceled", "Sam")
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    assert email.subject == "Your AutoGPT trial cancellation is confirmed"
    assert email.preheader == "Your card won't be charged."
    line = (
        f"You keep full access until {data.ends_label}. "
        "Resume your trial any time before then in billing settings."
    )
    for body in (email.html, email.text):
        assert line in body
        assert "immediately" not in body


def test_resumed_notice_copy_is_unchanged(trial, urls):
    data = notices.trial_notice_data(trial, "resumed", "Sam")
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    assert email.subject == "Your AutoGPT trial will continue into your plan"
    assert email.preheader == (
        "Your saved card will be charged for your Pro plan on "
        f"{data.ends_label} at $20.00 USD / month."
    )
    assert (
        "Applicable tax may be added. Review your plan or cancel in billing "
        f"settings before {data.ends_label}."
    ) in email.text
    assert "You keep full access" not in email.text


def _trial_subscription(trial: TrialState, **over) -> dict:
    return {
        "id": trial.subscription_id,
        "customer": trial.customer_id,
        "metadata": {"trial_enrollment_id": trial.id, "user_id": trial.user_id},
        **over,
    }


async def _on_update(trial: TrialState, previous: dict, **payload):
    with (
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial)
            ),
        ),
        patch.object(notices, "notify_trial", AsyncMock(return_value=True)) as notify,
    ):
        handled = await notices.on_trial_subscription_updated(
            _trial_subscription(trial, **payload), previous
        )
    return handled, notify


@pytest.mark.asyncio
@pytest.mark.parametrize("pending, kind", [(True, "canceled"), (False, "resumed")])
async def test_a_cancel_flip_sends_the_notice_for_the_current_state(
    trial, pending, kind
):
    trial.cancel_at_period_end = pending
    handled, notify = await _on_update(trial, {"cancel_at_period_end": not pending})
    assert handled
    notify.assert_awaited_once_with(_trial_subscription(trial), kind)


@pytest.mark.asyncio
async def test_an_unrelated_trial_update_sends_nothing(trial):
    handled, notify = await _on_update(trial, {"items": {"data": []}})
    assert handled
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_converting_early_from_a_pending_cancellation_is_not_a_resume(trial):
    """Subscribe now ends the trial and clears its pending cancellation in one
    update. The conversion notice covers it; no paid "resumed" email."""
    converted = trial.model_copy(update={"converted_at": datetime.now(UTC)})
    handled, notify = await _on_update(
        converted, {"cancel_at_period_end": True, "status": "trialing"}
    )
    assert handled
    notify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending, previous",
    [
        (True, {"cancel_at_period_end": False, "canceled_at": None}),
        (False, {"cancel_at_period_end": True, "canceled_at": 1789100000}),
    ],
    ids=["cancel", "resume"],
)
async def test_a_trial_era_flip_handled_after_conversion_is_not_paid_news(
    trial, pending, previous
):
    """A cancel or resume made during the trial, retried or delivered after
    the conversion, still describes the trial: no paid email, no notice."""
    converted = trial.model_copy(update={"converted_at": datetime.now(UTC)})
    handled, notify = await _on_update(
        converted, previous, status="trialing", cancel_at_period_end=pending
    )
    assert handled
    notify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "previous",
    [{"cancel_at_period_end": False}, {"status": "trialing"}, {"items": {}}],
)
async def test_a_converted_trial_is_left_to_the_paid_lifecycle(trial, previous):
    converted = trial.model_copy(update={"converted_at": datetime.now(UTC)})
    handled, notify = await _on_update(converted, previous)
    assert not handled
    notify.assert_not_awaited()


def test_an_early_conversion_seen_before_its_invoice_is_not_a_resume(trial):
    """If the conversion's update lands before the trial is marked converted,
    the flip routes to "resumed", which no longer applies: Stripe already
    ended the trial."""
    raw = {
        "status": "active",
        "trial_end": int(datetime.now(UTC).timestamp()),
        "cancel_at_period_end": False,
    }
    assert not notices._notice_applies(trial, "resumed", raw)
