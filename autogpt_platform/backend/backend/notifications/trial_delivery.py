"""RabbitMQ wakes durable trial deliveries; the database owns recovery."""

import asyncio
import logging

from backend.data.db_accessors import credit_db
from backend.data.trial_notifications import (
    ClaimedTrialDelivery,
    TrialDeliveryFinishStatus,
    TrialDeliveryMessage,
)
from backend.notifications.queue import queue_trial_delivery
from backend.notifications.trial import trial_notice_disposition
from backend.notifications.trial_postmark import TrialEmailSender
from backend.notifications.trial_recovery import recover_missing_trial_notices
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)
DELIVERY_TIMEOUT_SECONDS = 240


async def deliver_trial_notification(message: str) -> bool:
    wakeup = TrialDeliveryMessage.model_validate_json(message)
    delivery = await credit_db().claim_trial_notification(wakeup.delivery_id)
    if delivery is None:
        return True
    try:
        async with asyncio.timeout(DELIVERY_TIMEOUT_SECONDS):
            await _deliver_claimed(delivery, TrialEmailSender())
    except Exception as exc:
        await credit_db().retry_trial_notification(
            delivery.id, delivery.lease_token, type(exc).__name__
        )
        logger.exception("Trial notice %s deferred for durable retry", delivery.id)
    return True


async def _deliver_claimed(
    delivery: ClaimedTrialDelivery, sender: TrialEmailSender
) -> None:
    if delivery.attempts > 1:
        accepted = await sender.find_accepted(delivery.id)
        if accepted:
            await _finish(delivery, "accepted", accepted)
            return
    disposition = await trial_notice_disposition(delivery.user_id, delivery.payload)
    if disposition != "current":
        await _finish(delivery, disposition)
        return
    db = get_database_manager_async_client(should_retry=False)
    preference = await db.get_user_notification_preference(delivery.user_id)
    verified = await db.get_user_email_verification(delivery.user_id)
    if not preference.email or not verified:
        raise ValueError("Trial notice requires a verified recipient")
    accepted = await sender.send(delivery.id, preference.email, delivery.payload)
    await _finish(delivery, "accepted", accepted)


async def _finish(
    delivery: ClaimedTrialDelivery,
    status: TrialDeliveryFinishStatus,
    message_id: str | None = None,
) -> None:
    finished = await credit_db().finish_trial_notification(
        delivery.id, delivery.lease_token, status, message_id
    )
    if not finished:
        raise RuntimeError("Trial notification delivery lease is no longer owned")


async def recover_trial_notifications() -> None:
    try:
        await recover_missing_trial_notices()
    finally:
        await _wake_due_deliveries()


async def _wake_due_deliveries() -> None:
    for delivery_id in await credit_db().get_due_trial_notifications():
        result = await queue_trial_delivery(delivery_id)
        if not result.success:
            raise RuntimeError("Trial notification wake-up queue is unavailable")
        await credit_db().mark_trial_notification_queued(delivery_id)
