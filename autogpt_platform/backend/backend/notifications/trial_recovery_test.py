from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.trial_notification_recovery import TrialNoticeCandidate
from backend.notifications import trial_recovery as recovery
from backend.notifications import trial_test as fixtures

trial = fixtures.trial


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, ["started"]),
        ({"ends_at": datetime.now(UTC) + timedelta(days=2)}, ["started", "ending"]),
        ({"cancel_at_period_end": True, "notification_revision": 1}, ["canceled"]),
        ({"notification_revision": 2}, ["started", "resumed"]),
        ({"status": "canceled"}, ["ended"]),
        ({"status": "paused"}, ["ended"]),
        ({"status": "past_due"}, ["payment_failed"]),
        ({"status": "unpaid"}, ["payment_failed"]),
        (
            {
                "status": "active",
                "converted_at": datetime.now(UTC),
                "conversion_invoice_id": "in_first",
            },
            ["converted"],
        ),
    ],
)
def test_repair_selects_the_current_customer_notice(trial, changes, expected):
    assert (
        recovery.due_trial_notices(trial.model_copy(update=changes), datetime.now(UTC))
        == expected
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"consumed_at": None},
        {"ends_at": None},
        {"ends_at": datetime.now(UTC) - timedelta(seconds=1)},
        {"card_verified_at": None},
        {"status": "canceled", "converted_at": datetime.now(UTC)},
    ],
)
def test_repair_does_not_invent_a_trial_or_replay_paid_cancellation(trial, changes):
    assert not recovery.due_trial_notices(
        trial.model_copy(update=changes), datetime.now(UTC)
    )


@pytest.mark.asyncio
async def test_repair_refreshes_via_database_rpc_before_emitting(trial):
    candidate = TrialNoticeCandidate(
        id=trial.id, user_id=trial.user_id, subscription_id=trial.subscription_id
    )
    database = MagicMock(
        get_trial_notice_candidates=AsyncMock(side_effect=[[candidate], []]),
        sync_subscription_from_stripe=AsyncMock(),
        get_subscription_trial=AsyncMock(return_value=trial),
    )
    subscription = {"id": trial.subscription_id}
    with (
        patch.object(recovery, "credit_db", return_value=database),
        patch.object(recovery, "stripe_call", AsyncMock(return_value=subscription)),
        patch.object(recovery, "notify_trial", AsyncMock()) as notify,
    ):
        await recovery.recover_missing_trial_notices()
    database.sync_subscription_from_stripe.assert_awaited_once_with(subscription)
    notify.assert_awaited_once_with(subscription, "started")
    assert database.get_trial_notice_candidates.await_args_list[1].args == (trial.id,)


@pytest.mark.asyncio
async def test_bad_candidate_does_not_starve_later_pages(trial):
    bad = TrialNoticeCandidate(id="a", user_id="bad", subscription_id="sub_bad")
    good = TrialNoticeCandidate(
        id=trial.id, user_id=trial.user_id, subscription_id=trial.subscription_id
    )
    database = MagicMock(
        get_trial_notice_candidates=AsyncMock(side_effect=[[bad], [good], []]),
        sync_subscription_from_stripe=AsyncMock(),
        get_subscription_trial=AsyncMock(return_value=trial),
    )
    with (
        patch.object(recovery, "credit_db", return_value=database),
        patch.object(
            recovery,
            "stripe_call",
            AsyncMock(side_effect=[TimeoutError(), {"id": trial.subscription_id}]),
        ),
        patch.object(recovery, "notify_trial", AsyncMock()) as notify,
    ):
        await recovery.recover_missing_trial_notices()
    notify.assert_awaited_once()
    assert database.get_trial_notice_candidates.await_args_list[1].args == ("a",)
