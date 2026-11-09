from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import NotificationEventModel, TrialUpdateData
from backend.notifications import notifications as delivery
from backend.notifications import trial_test as fixtures
from backend.notifications.queue import create_notification_config, get_routing_key
from backend.notifications.trial import trial_notice_data, trial_notice_key

trial = fixtures.trial


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current,verified,fail",
    [
        (True, True, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
async def test_trial_mail_uses_shared_sender_and_preserves_delivery_gates(
    trial, db_client, current, verified, fail
):
    data = trial_notice_data(trial, "started", "Sam")
    data.notice_key = trial_notice_key(trial, "started")
    event = NotificationEventModel[TrialUpdateData](
        user_id=trial.user_id, type=NotificationType.TRIAL_UPDATE, data=data
    )
    manager = delivery.NotificationManager.__new__(delivery.NotificationManager)
    sender = AsyncMock(
        side_effect=RuntimeError("provider unavailable") if fail else None
    )
    manager.email_sender = MagicMock(send_notification=sender)
    db_client.get_user_email_verification.return_value = verified
    db_client.get_user_notification_preference.return_value.daily_limit = 0
    with (
        patch.object(
            delivery,
            "trial_notice_disposition",
            AsyncMock(return_value="current" if current else "obsolete"),
        ),
        patch.object(
            delivery, "get_database_manager_async_client", return_value=db_client
        ),
        patch.object(
            delivery,
            "generate_unsubscribe_link",
            return_value="https://example.com/prefs",
        ),
        patch.object(
            delivery,
            "generate_preference_link",
            return_value="https://example.com/prefs",
        ),
        patch.object(
            delivery, "claim_daily_send", AsyncMock(return_value=False)
        ) as cap,
    ):
        if fail:
            with pytest.raises(RuntimeError, match="provider unavailable"):
                await manager._process_user_notification(event.model_dump_json())
        else:
            assert await manager._process_user_notification(event.model_dump_json())
    assert sender.await_count == int(current and verified)
    cap.assert_not_awaited()


def test_trial_notices_have_no_separate_queue():
    assert (
        get_routing_key(NotificationType.TRIAL_UPDATE)
        == "notification.user.TRIAL_UPDATE"
    )
    assert all(
        "trial" not in queue.name for queue in create_notification_config().queues
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recovers", [True, False])
async def test_suppressed_trial_mail_uses_shared_retry_and_dlq(
    trial, db_client, recovers
):
    data = trial_notice_data(trial, "started", "Sam")
    data.notice_key = trial_notice_key(trial, "started")
    event = NotificationEventModel[TrialUpdateData](
        user_id=trial.user_id, type=NotificationType.TRIAL_UPDATE, data=data
    )
    manager = delivery.NotificationManager.__new__(delivery.NotificationManager)
    sender = AsyncMock()
    manager.email_sender = MagicMock(send_notification=sender)
    message = MagicMock(
        body=event.model_dump_json().encode(), ack=AsyncMock(), reject=AsyncMock()
    )
    dispositions = (
        ["suppressed", "current"]
        if recovers
        else ["suppressed"] * delivery.MAX_CONSUMER_RETRY_ATTEMPTS
    )
    with (
        patch.object(
            delivery, "trial_notice_disposition", AsyncMock(side_effect=dispositions)
        ),
        patch.object(
            delivery, "get_database_manager_async_client", return_value=db_client
        ),
        patch.object(
            delivery,
            "generate_unsubscribe_link",
            return_value="https://example.com/prefs",
        ),
        patch.object(
            delivery,
            "generate_preference_link",
            return_value="https://example.com/prefs",
        ),
        patch.object(delivery.asyncio, "sleep", AsyncMock()) as sleep,
        patch("backend.notifications.trial.release_claim", AsyncMock()) as release,
        patch(
            "backend.notifications.trial.queue_notification_async", AsyncMock()
        ) as publish,
    ):
        await manager._process_message_with_retry(
            message, manager._process_user_notification, "user_notifications_v3"
        )
    assert sender.await_count == int(recovers)
    assert message.ack.await_count == int(recovers)
    if not recovers:
        message.reject.assert_awaited_once_with(requeue=False)
    assert sleep.await_count > 0
    release.assert_not_awaited()
    publish.assert_not_awaited()
