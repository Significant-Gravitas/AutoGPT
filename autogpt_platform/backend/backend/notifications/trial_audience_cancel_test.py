"""The MailerLite trial group across a scheduled cancellation: each cancel or
resume flip moves the group once, a pending cancellation that runs out leaves
it, a trial ended by buying another plan still leaves the group, and while it
is pending the ending reminder is the not-renewing one, which moves no group. The status itself is the lifecycle
sync's (`lifecycle_audience.py`)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.data.notifications import AudienceAction, NotificationResult
from backend.notifications import trial as notices
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
    assert [e.action for e in events] == [
        AudienceAction.REMOVE_TRIAL,
        AudienceAction.ADD_TRIAL,
    ]


@pytest.mark.asyncio
async def test_a_cancel_pending_trial_that_runs_out_leaves_the_group(trial):
    pending = trial.model_copy(
        update={"cancel_at_period_end": True, "status": "canceled"}
    )
    pending, raw = _state(pending, "ended")
    raw["cancel_at_period_end"] = True
    audience = AsyncMock(return_value=NotificationResult(success=True))
    got, notice, _ = await _notify(pending, raw, "ended", audience=audience)
    assert got == [AudienceAction.REMOVE_TRIAL]
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_ending_event_while_cancellation_pending_sends_not_renewing_notice(
    trial,
):
    pending, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": True}), "canceled"
    )
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=3)).timestamp())
    claim = AsyncMock(return_value=True)
    got, notice, _ = await _notify(pending, raw, "ending", claim_once=claim)
    assert got == []
    notice.assert_awaited_once()
    assert notice.await_args.args[0].data.kind == "expiring"
    [key] = [c.args[0] for c in claim.await_args_list]
    assert key.startswith("trial:trial-1:expiring:")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,left", [("ending", 3), ("last_call", 1)])
async def test_no_not_renewing_notice_while_another_plan_is_live(trial, kind, left):
    """A newer plan keeps access past the trial's end, so "your access ends"
    would be wrong."""
    pending, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": True}), "canceled"
    )
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=left)).timestamp())
    claim = AsyncMock(return_value=True)
    got, notice, release = await _notify(
        pending, raw, kind, others=("sub_max",), claim_once=claim
    )
    assert got == []
    notice.assert_not_awaited()
    # Claimed first so a repeat pass stops early, then given back.
    assert claim.await_count == release.await_count == 1


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


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed", [True, False])
async def test_already_sent_last_call_skips_the_plan_listing(trial, claimed):
    """The hourly pass sees a due trial on several runs; once its notice is
    claimed, later runs stop before listing the customer's plans."""
    pending, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": True}), "canceled"
    )
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=1)).timestamp())
    outlives = AsyncMock(return_value=False)
    with patch.object(notices, "_another_plan_outlives", outlives):
        _, notice, _ = await _notify(pending, raw, "last_call", claimed=claimed)
    assert outlives.await_count == int(claimed)
    assert notice.await_count == int(claimed)


@pytest.mark.asyncio
async def test_claim_is_given_back_when_another_plan_outlives_the_trial(trial):
    pending, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": True}), "canceled"
    )
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=1)).timestamp())
    _, notice, release = await _notify(pending, raw, "last_call", others=("sub_max",))
    notice.assert_not_awaited()
    release.assert_awaited_once_with(notices.trial_notice_key(pending, "last_call"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,left,fail_open", [("expiring", 3, True), ("last_call", 1, False)]
)
async def test_only_the_last_call_claim_fails_closed(trial, kind, left, fail_open):
    pending, raw = _state(
        trial.model_copy(update={"cancel_at_period_end": True}), "canceled"
    )
    raw["trial_end"] = int((datetime.now(UTC) + timedelta(days=left)).timestamp())
    claim = AsyncMock(return_value=True)
    await _notify(
        pending, raw, "ending" if kind == "expiring" else kind, claim_once=claim
    )
    assert claim.await_args.kwargs == {"fail_open": fail_open}
