from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import NotificationResult
from backend.notifications import trial as notices
from backend.notifications import trial_test as fixtures

trial = fixtures.trial


def subscription(trial):
    return {
        "id": trial.subscription_id,
        "customer": trial.customer_id,
        "status": trial.status,
        "trial_end": int(trial.ends_at.timestamp()),
        "cancel_at_period_end": trial.cancel_at_period_end,
        "metadata": {
            "user_id": trial.user_id,
            "trial_enrollment_id": trial.id,
            "trial_checkout_attempt": str(trial.checkout_attempt),
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"card_verified_at": None}, "suppressed"),
        ({"ends_at": datetime.now(UTC) + timedelta(days=8)}, "obsolete"),
        ({"notification_revision": 4}, "obsolete"),
    ],
)
async def test_delivery_reconciles_and_suppresses_outdated_notice(
    trial, changes, expected
):
    trial.notification_revision = 2
    data = notices.trial_notice_data(trial, "resumed", "Sam").model_copy(
        update={"notice_key": notices.trial_notice_key(trial, "resumed")}
    )
    fresh = trial.model_copy(update=changes)
    database = MagicMock(
        get_subscription_trial=AsyncMock(side_effect=[trial, fresh]),
        sync_subscription_from_stripe=AsyncMock(),
    )
    raw = subscription(fresh)
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
    ):
        assert await notices.trial_notice_disposition(trial.user_id, data) == expected
    database.sync_subscription_from_stripe.assert_awaited_once_with(raw)


@pytest.mark.asyncio
async def test_old_checkout_attempt_is_acknowledged_without_a_notice(trial):
    stale = subscription(trial)
    trial = trial.model_copy(
        update={"checkout_attempt": 1, "subscription_id": "sub_new"}
    )
    database = MagicMock(get_subscription_trial=AsyncMock(return_value=trial))
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(notices, "stripe_call", AsyncMock(return_value=stale)),
        patch.object(notices, "queue_notification_async", AsyncMock()) as queue,
    ):
        assert await notices.notify_trial(stale, "started")
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconsumed_checkout_without_end_date_does_not_publish(trial):
    raw = subscription(trial)
    raw["trial_end"] = None
    pending = trial.model_copy(update={"consumed_at": None, "ends_at": None})
    database = MagicMock(get_subscription_trial=AsyncMock(return_value=pending))
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
        patch.object(notices, "queue_notification_async", AsyncMock()) as queue,
    ):
        assert await notices.notify_trial(raw, "started")
    queue.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,changes",
    [
        ("started", {}),
        ("ending", {"ends_at": datetime.now(UTC) + timedelta(days=2)}),
        ("canceled", {"cancel_at_period_end": True, "notification_revision": 1}),
        ("resumed", {"notification_revision": 2}),
        ("ended", {"status": "canceled"}),
        ("payment_failed", {"status": "past_due"}),
        (
            "converted",
            {
                "status": "active",
                "converted_at": datetime.now(UTC),
                "conversion_invoice_id": "in_first",
            },
        ),
    ],
)
async def test_current_notice_survives_authoritative_refresh(trial, kind, changes):
    trial = trial.model_copy(update=changes)
    data = notices.trial_notice_data(trial, kind, "Sam")
    data.notice_key = notices.trial_notice_key(trial, kind)
    database = MagicMock(
        get_subscription_trial=AsyncMock(return_value=trial),
        sync_subscription_from_stripe=AsyncMock(),
    )
    raw = subscription(trial)
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
    ):
        assert await notices.trial_notice_disposition(trial.user_id, data) == "current"
    database.sync_subscription_from_stripe.assert_awaited_once_with(raw)


@pytest.mark.asyncio
async def test_unidentified_legacy_payload_cannot_bypass_freshness_check(trial):
    data = notices.trial_notice_data(trial, "started", "Sam")
    with patch.object(notices, "credit_db") as database:
        assert await notices.trial_notice_disposition(trial.user_id, data) == "obsolete"
    database.assert_not_called()


@pytest.mark.asyncio
async def test_state_change_during_refresh_does_not_send_old_welcome(trial):
    data = notices.trial_notice_data(trial, "started", "Sam")
    data.notice_key = notices.trial_notice_key(trial, "started")
    fresh = trial.model_copy(update={"cancel_at_period_end": True})
    database = MagicMock(
        get_subscription_trial=AsyncMock(side_effect=[trial, fresh]),
        sync_subscription_from_stripe=AsyncMock(),
    )
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(
            notices, "stripe_call", AsyncMock(return_value=subscription(trial))
        ),
    ):
        assert (
            await notices.trial_notice_disposition(trial.user_id, data) == "suppressed"
        )


@pytest.mark.asyncio
async def test_delivery_marks_changed_terms_obsolete_without_sending(trial):
    data = notices.trial_notice_data(trial, "started", "Sam")
    data.notice_key = notices.trial_notice_key(trial, "started")
    data.ends_label = "Obsolete end date"
    database = MagicMock(
        get_subscription_trial=AsyncMock(return_value=trial),
        sync_subscription_from_stripe=AsyncMock(),
    )
    with (
        patch.object(notices, "credit_db", return_value=database),
        patch.object(
            notices, "stripe_call", AsyncMock(return_value=subscription(trial))
        ),
    ):
        assert await notices.trial_notice_disposition(trial.user_id, data) == "obsolete"


async def _notify_ended(trial, others: dict[str, list[str]]):
    """`others` maps a Stripe status to the customer's subscription ids in it;
    the list double honours the status filter the way Stripe does."""
    raw = subscription(trial)

    async def stripe_call(fn, *args, **kwargs):
        if fn != notices.stripe.Subscription.list_async:
            return raw
        assert kwargs["customer"] == trial.customer_id
        status = kwargs["status"]
        ids = [
            sub_id
            for key, subs in others.items()
            if status in (key, "all")
            for sub_id in subs
        ]
        return SimpleNamespace(
            data=[SimpleNamespace(id=sub_id) for sub_id in ids], has_more=False
        )

    user = SimpleNamespace(id=trial.user_id, name="Sam", email="sam@example.com")
    with (
        patch.object(notices, "stripe_call", AsyncMock(side_effect=stripe_call)),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial)
            ),
        ),
        patch.object(
            notices,
            "user_db",
            return_value=MagicMock(get_user_by_id=AsyncMock(return_value=user)),
        ),
        patch.object(notices, "claim_once", AsyncMock(return_value=True)) as claim,
        patch.object(notices, "queue_trial_audience_change", AsyncMock()) as audience,
        patch.object(notices, "leave_trial_group", AsyncMock()) as leave,
        patch.object(
            notices,
            "queue_notification_async",
            AsyncMock(return_value=NotificationResult(success=True)),
        ) as queue,
        patch.object(notices, "_track_billing_event") as track,
    ):
        handled = await notices.notify_trial(raw, "ended")
    return handled, SimpleNamespace(
        claim=claim, audience=audience, leave=leave, queue=queue, track=track, user=user
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "others",
    [{"active": ["sub_max"]}, {"trialing": ["sub_other"]}],
    ids=["active-plan", "trialing-plan"],
)
async def test_a_trial_ended_by_buying_another_plan_sends_nothing(trial, others):
    """Buying another plan while cancel-pending ends the trial subscription.
    No "trial ended" email, no trial_canceled overwrite of the new plan's
    MailerLite status, no trial_ended event. The person still leaves the
    MailerLite trial group."""
    trial.status = "canceled"
    trial.cancel_at_period_end = True
    handled, sent = await _notify_ended(trial, others)
    assert handled
    sent.leave.assert_awaited_once_with(sent.user)
    sent.claim.assert_not_awaited()
    sent.audience.assert_not_awaited()
    sent.queue.assert_not_awaited()
    sent.track.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "others",
    [
        {},
        {"active": ["sub_1"], "trialing": ["sub_1"]},
        {"canceled": ["sub_old"], "past_due": ["sub_due"]},
    ],
    ids=["no-other-plan", "only-itself", "none-live"],
)
async def test_a_cancel_pending_trial_reaching_its_end_sends_the_ended_notice(
    trial, others
):
    trial.status = "canceled"
    trial.cancel_at_period_end = True
    handled, sent = await _notify_ended(trial, others)
    assert handled
    sent.leave.assert_not_awaited()
    sent.audience.assert_awaited_once()
    assert sent.audience.await_args.args[0] == "ended"
    sent.queue.assert_awaited_once()
    assert sent.queue.await_args.args[0].data.kind == "ended"
    assert sent.track.call_args.args[0] == "trial_ended"
