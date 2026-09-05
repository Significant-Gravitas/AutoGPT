import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import (
    NotificationResult,
    SubscriptionPlan,
    TrialUpdateData,
)
from backend.data.trial_notifications import ClaimedTrialDelivery, TrialDeliveryMessage
from backend.notifications import trial_delivery as worker


@pytest.fixture
def delivery():
    return ClaimedTrialDelivery(
        id="notice-1",
        user_id="user-1",
        trial_id="trial-1",
        attempts=1,
        lease_token="lease-1",
        created_at=datetime.now(UTC),
        payload=TrialUpdateData(
            user_name="Sam",
            kind="started",
            ends_label="17 Sep 2026",
            onboarding_credit_amount=300,
            offer_version="offer-v1",
            plan=SubscriptionPlan(
                name="Pro",
                cycle="monthly",
                cycle_noun="month",
                label="Pro",
                price_display="$20.00 / month",
            ),
        ),
    )


@pytest.fixture
def boundaries(delivery):
    db = MagicMock(
        claim_trial_notification=AsyncMock(return_value=delivery),
        finish_trial_notification=AsyncMock(return_value=True),
        retry_trial_notification=AsyncMock(),
        get_due_trial_notifications=AsyncMock(return_value=[delivery.id]),
        mark_trial_notification_queued=AsyncMock(),
        get_user_notification_preference=AsyncMock(
            return_value=MagicMock(email="sam@example.com")
        ),
        get_user_email_verification=AsyncMock(return_value=True),
    )
    sender = MagicMock(
        send=AsyncMock(return_value="message-1"),
        find_accepted=AsyncMock(return_value=None),
    )
    with (
        patch.object(worker, "credit_db", return_value=db),
        patch.object(worker, "get_database_manager_async_client", return_value=db),
        patch.object(worker, "TrialEmailSender", return_value=sender),
        patch.object(
            worker, "trial_notice_is_current", AsyncMock(return_value=True)
        ) as current,
    ):
        yield db, sender, current


async def deliver():
    return await worker.deliver_trial_notification(
        TrialDeliveryMessage(delivery_id="notice-1").model_dump_json()
    )


@pytest.mark.asyncio
async def test_success_records_provider_acceptance(boundaries, delivery):
    db, sender, _ = boundaries
    assert await deliver()
    sender.send.assert_awaited_once_with(
        delivery.id, "sam@example.com", delivery.payload
    )
    db.finish_trial_notification.assert_awaited_once_with(
        "notice-1", "lease-1", "accepted", "message-1"
    )


@pytest.mark.asyncio
async def test_another_worker_or_terminal_notice_never_sends(boundaries):
    db, sender, _ = boundaries
    db.claim_trial_notification.return_value = None
    assert await deliver()
    sender.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_acceptance_after_crash_is_reconciled_without_resend(
    boundaries, delivery
):
    db, sender, _ = boundaries
    delivery.attempts = 2
    sender.find_accepted.return_value = "accepted-before-crash"
    assert await deliver()
    sender.send.assert_not_awaited()
    db.finish_trial_notification.assert_awaited_once_with(
        "notice-1", "lease-1", "accepted", "accepted-before-crash"
    )


@pytest.mark.asyncio
async def test_provider_lookup_failure_defers_instead_of_blind_resend(
    boundaries, delivery
):
    db, sender, _ = boundaries
    delivery.attempts = 2
    sender.find_accepted.side_effect = RuntimeError("provider unavailable")
    assert await deliver()
    sender.send.assert_not_awaited()
    db.retry_trial_notification.assert_awaited_once_with(
        "notice-1", "lease-1", "RuntimeError"
    )


@pytest.mark.asyncio
async def test_stale_notice_is_suppressed(boundaries):
    db, sender, current = boundaries
    current.return_value = False
    assert await deliver()
    sender.send.assert_not_awaited()
    db.finish_trial_notification.assert_awaited_once_with(
        "notice-1", "lease-1", "suppressed", None
    )


@pytest.mark.asyncio
async def test_unverified_recipient_is_retryable_not_marked_sent(boundaries):
    db, sender, _ = boundaries
    db.get_user_email_verification.return_value = False
    assert await deliver()
    sender.send.assert_not_awaited()
    db.finish_trial_notification.assert_not_awaited()
    db.retry_trial_notification.assert_awaited_once()


@pytest.mark.asyncio
async def test_database_failure_during_retry_does_not_ack_the_wakeup(boundaries):
    db, sender, _ = boundaries
    sender.send.side_effect = RuntimeError("timeout")
    db.retry_trial_notification.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await deliver()


@pytest.mark.asyncio
async def test_delivery_deadline_is_shorter_than_the_lease(boundaries):
    db, sender, _ = boundaries

    async def delayed_send(*args):
        await asyncio.sleep(1)
        return "too-late"

    sender.send.side_effect = delayed_send
    with patch.object(worker, "DELIVERY_TIMEOUT_SECONDS", 0.01):
        assert await deliver()
    db.finish_trial_notification.assert_not_awaited()
    db.retry_trial_notification.assert_awaited_once_with(
        "notice-1", "lease-1", "TimeoutError"
    )


@pytest.mark.asyncio
async def test_recovery_publishes_durable_ids_and_delays_only_future_wakeups(
    boundaries,
):
    db, _, _ = boundaries
    with patch.object(
        worker,
        "queue_trial_delivery",
        AsyncMock(return_value=NotificationResult(success=True)),
    ) as queue:
        await worker.recover_trial_notifications()
    queue.assert_awaited_once_with("notice-1")
    db.mark_trial_notification_queued.assert_awaited_once_with("notice-1")


@pytest.mark.asyncio
async def test_queue_outage_leaves_notice_due_for_next_recovery(boundaries):
    db, _, _ = boundaries
    with patch.object(
        worker,
        "queue_trial_delivery",
        AsyncMock(return_value=NotificationResult(success=False)),
    ):
        with pytest.raises(RuntimeError, match="unavailable"):
            await worker.recover_trial_notifications()
    db.mark_trial_notification_queued.assert_not_awaited()
