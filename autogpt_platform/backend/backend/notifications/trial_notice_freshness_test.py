from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
