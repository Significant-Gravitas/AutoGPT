"""The MailerLite trial group across a scheduled cancellation: each cancel or
resume flip moves the group once, a pending cancellation that runs out leaves
it as trial canceled, a trial ended by buying another plan still leaves the
group without touching that plan's status, and no ending reminder goes out
while it is pending."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from backend.data.notifications import (
    AudienceAction,
    NotificationResult,
    SubscriberField,
    SubscriptionStatus,
)
from backend.notifications import trial_audience_test
from backend.notifications.trial_audience_test import _notify, _state

trial = trial_audience_test.trial


def _claims(keys: set[str]) -> AsyncMock:
    """Redis SET NX: each key is claimed once."""

    async def claim(key: str) -> bool:
        fresh = key not in keys
        keys.add(key)
        return fresh

    return AsyncMock(side_effect=claim)


async def _flip(trial, kind, revision, keys, audience):
    state, raw = _state(
        trial.model_copy(
            update={
                "cancel_at_period_end": kind == "canceled",
                "notification_revision": revision,
            }
        ),
        kind,
    )
    _, notice, _ = await _notify(
        state, raw, kind, audience=audience, claim_once=_claims(keys)
    )
    return notice.await_count


@pytest.mark.asyncio
async def test_a_second_cancel_at_the_same_revision_queues_once(trial):
    keys: set[str] = set()
    audience = AsyncMock(return_value=NotificationResult(success=True))
    sent = [await _flip(trial, "canceled", rev, keys, audience) for rev in (3, 3, 4)]
    assert sent == [1, 0, 1]
    assert [c.args[0].action for c in audience.await_args_list] == [
        AudienceAction.REMOVE_TRIAL,
        AudienceAction.REMOVE_TRIAL,
    ]
    assert keys == {"trial:trial-1:canceled:3", "trial:trial-1:canceled:4"}


@pytest.mark.asyncio
async def test_each_cancel_or_resume_flip_moves_the_group_once(trial):
    """Each flip bumps the revision once, so a replayed webhook for the same
    flip changes nothing and the next flip gets its own notice."""
    keys: set[str] = set()
    audience = AsyncMock(return_value=NotificationResult(success=True))
    flips = [("canceled", 1), ("canceled", 1), ("resumed", 2), ("resumed", 2)]
    sent = [await _flip(trial, kind, rev, keys, audience) for kind, rev in flips]
    assert sent == [1, 0, 1, 0]
    events = [c.args[0] for c in audience.await_args_list]
    assert [(e.action, e.fields[SubscriberField.STATUS]) for e in events] == [
        (AudienceAction.REMOVE_TRIAL, SubscriptionStatus.TRIAL_CANCELED.value),
        (AudienceAction.ADD_TRIAL, SubscriptionStatus.IN_TRIAL.value),
    ]


@pytest.mark.asyncio
async def test_a_cancel_pending_trial_that_runs_out_leaves_the_group_as_trial_canceled(
    trial,
):
    pending = trial.model_copy(
        update={"cancel_at_period_end": True, "status": "canceled"}
    )
    pending, raw = _state(pending, "ended")
    raw["cancel_at_period_end"] = True
    audience = AsyncMock(return_value=NotificationResult(success=True))
    got, notice, _ = await _notify(pending, raw, "ended", audience=audience)
    assert got == [AudienceAction.REMOVE_TRIAL]
    status = audience.await_args.args[0].fields[SubscriberField.STATUS]
    assert status == SubscriptionStatus.TRIAL_CANCELED.value
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_ending_reminder_while_the_cancellation_is_pending(trial):
    pending, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": True}), "canceled"
    )
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=1)).timestamp())
    claim = AsyncMock(return_value=True)
    got, notice, _ = await _notify(pending, raw, "ending", claim_once=claim)
    assert got == []
    notice.assert_not_awaited()
    claim.assert_not_awaited()


async def _end_under_another_plan(trial, pending: bool, **kwargs):
    ended, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": pending}), "ended"
    )
    raw["cancel_at_period_end"] = pending
    claim = AsyncMock(return_value=True)
    got, notice, release = await _notify(
        ended, raw, "ended", others=("sub_max",), claim_once=claim, **kwargs
    )
    claim.assert_not_awaited()
    notice.assert_not_awaited()
    release.assert_not_awaited()
    return got


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending", [True, False], ids=["cancel-pending", "ended-immediately"]
)
async def test_a_trial_ended_by_another_plan_still_leaves_the_trial_group(
    trial, pending
):
    """Its "ended" notice is not sent and the new plan's status must stay,
    but the person is no longer on a trial. That holds whether the trial was
    set to cancel at its end or (flag off) ended at once."""
    audience = AsyncMock(return_value=NotificationResult(success=True))
    got = await _end_under_another_plan(trial, pending, audience=audience)
    assert got == [AudienceAction.REMOVE_TRIAL]
    assert audience.await_args.args[0].fields == {}


@pytest.mark.asyncio
async def test_an_opted_out_trialist_ended_by_another_plan_queues_nothing(trial):
    got = await _end_under_another_plan(trial, True, opted_out_at=datetime.now(UTC))
    assert got == []


@pytest.mark.asyncio
async def test_a_group_exit_that_cannot_be_queued_fails_so_stripe_retries(trial):
    failing = AsyncMock(return_value=NotificationResult(success=False, message="down"))
    got = await _end_under_another_plan(trial, True, audience=failing, raises=True)
    assert got == [AudienceAction.REMOVE_TRIAL]
