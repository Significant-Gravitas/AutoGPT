"""RabbitMQ wiring for notifications.

Three destinations, because there are three genuinely different consumers:
customer-facing mail, internal ops mail, and MailerLite audience changes. The
old batch and summary queues are gone with the per-run email and the summary
digests they carried — the Briefing is assembled on a schedule, not batched
from queued events.
"""

import logging

from prisma.enums import NotificationType

from backend.data.notifications import (
    AudienceEventModel,
    DeliveryStream,
    NotificationEventModel,
    NotificationResult,
    get_delivery_stream,
)
from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), "[NotificationQueue]")

NOTIFICATION_EXCHANGE = Exchange(name="notifications", type=ExchangeType.TOPIC)
DEAD_LETTER_EXCHANGE = Exchange(name="dead_letter", type=ExchangeType.TOPIC)
EXCHANGES = [NOTIFICATION_EXCHANGE, DEAD_LETTER_EXCHANGE]

# ``_v3`` marks the cutover to the redesigned email system: the old queues hold
# payloads in a shape no current template can render, so they are not reused.
USER_NOTIFICATIONS_QUEUE = "user_notifications_v3"
OPS_NOTIFICATIONS_QUEUE = "ops_notifications_v3"
AUDIENCE_QUEUE = "audience_changes_v3"
FAILED_NOTIFICATIONS_QUEUE = "failed_notifications_v3"

_QUORUM = "quorum"


def create_notification_config() -> RabbitMQConfig:
    queues = [
        _queue(USER_NOTIFICATIONS_QUEUE, "notification.user.#", "failed.user"),
        _queue(OPS_NOTIFICATIONS_QUEUE, "notification.ops.#", "failed.ops"),
        _queue(AUDIENCE_QUEUE, "notification.audience.#", "failed.audience"),
        # DLQ destination — quorum so dead letters survive a broker restart.
        Queue(
            name=FAILED_NOTIFICATIONS_QUEUE,
            exchange=DEAD_LETTER_EXCHANGE,
            routing_key="failed.#",
            arguments={"x-queue-type": _QUORUM},
        ),
    ]
    return RabbitMQConfig(exchanges=EXCHANGES, queues=queues)


def get_routing_key(event_type: NotificationType) -> str:
    stream = get_delivery_stream(event_type)
    destination = "ops" if stream is DeliveryStream.OPS else "user"
    return f"notification.{destination}.{event_type.value}"


async def queue_notification_async(
    event: NotificationEventModel,
) -> NotificationResult:
    """Hand a notification to the sender. Callers never send inline: the
    consumer owns preference checks, rendering and retries."""
    return await _publish(get_routing_key(event.type), event.model_dump_json())


async def queue_audience_change(event: AudienceEventModel) -> NotificationResult:
    """Hand a MailerLite audience change to the queue.

    Never call MailerLite inline from a Stripe webhook: an outage there must
    not fail payment processing.
    """
    return await _publish(
        f"notification.audience.{event.action.value}", event.model_dump_json()
    )


async def _publish(routing_key: str, message: str) -> NotificationResult:
    try:
        from backend.util.clients import get_async_notification_queue

        queue = await get_async_notification_queue()
        await queue.publish_message(
            routing_key=routing_key,
            message=message,
            exchange=NOTIFICATION_EXCHANGE,
        )
        return NotificationResult(
            success=True, message=f"Queued with routing key: {routing_key}"
        )
    except Exception as e:
        logger.exception(f"Error queueing {routing_key}: {e}")
        return NotificationResult(success=False, message=str(e))


def _queue(name: str, routing_key: str, dead_letter_key: str) -> Queue:
    return Queue(
        name=name,
        exchange=NOTIFICATION_EXCHANGE,
        routing_key=routing_key,
        arguments={
            "x-queue-type": _QUORUM,
            "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
            "x-dead-letter-routing-key": dead_letter_key,
        },
    )
