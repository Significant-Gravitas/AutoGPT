import json
import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from autogpt_libs.utils.cache import thread_cached

from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.executor.database import DatabaseManager
from backend.notifications.models import (
    BatchingStrategy,
    NotificationEvent,
    NotificationResult,
)
from backend.notifications.summary import SummaryManager
from backend.util.service import AppService, expose, get_service_client
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.executor import DatabaseManager

logger = logging.getLogger(__name__)
settings = Settings()


def create_notification_config() -> RabbitMQConfig:
    """Create RabbitMQ configuration for notifications"""
    notification_exchange = Exchange(name="notifications", type=ExchangeType.TOPIC)

    batch_exchange = Exchange(name="batching", type=ExchangeType.TOPIC)

    summary_exchange = Exchange(name="summaries", type=ExchangeType.TOPIC)

    dead_letter_exchange = Exchange(name="dead_letter", type=ExchangeType.DIRECT)

    queues = [
        # Main notification queues
        Queue(
            name="immediate_notifications",
            exchange=notification_exchange,
            routing_key="notification.immediate.#",
            # dead_letter_exchange=dead_letter_exchange,
            # dead_letter_routing_key="failed.immediate",
        ),
        Queue(
            name="backoff_notifications",
            exchange=notification_exchange,
            routing_key="notification.backoff.#",
            arguments={
                "x-dead-letter-exchange": dead_letter_exchange.name,
                "x-dead-letter-routing_key": "failed.backoff",
            },
        ),
        # Batch queues for aggregation
        Queue(
            name="hourly_batch", exchange=batch_exchange, routing_key="batch.hourly.#"
        ),
        Queue(name="daily_batch", exchange=batch_exchange, routing_key="batch.daily.#"),
        # Summary queues
        Queue(
            name="daily_summary_trigger",
            exchange=summary_exchange,
            routing_key="summary.daily",
            arguments={"x-message-ttl": 86400000},  # 24 hours
        ),
        Queue(
            name="weekly_summary_trigger",
            exchange=summary_exchange,
            routing_key="summary.weekly",
            arguments={"x-message-ttl": 604800000},  # 7 days
        ),
        Queue(
            name="monthly_summary_trigger",
            exchange=summary_exchange,
            routing_key="summary.monthly",
            arguments={"x-message-ttl": 2592000000},  # 30 days
        ),
        # Failed notifications queue
        Queue(
            name="failed_notifications",
            exchange=dead_letter_exchange,
            routing_key="failed.#",
        ),
    ]

    return RabbitMQConfig(
        exchanges=[
            notification_exchange,
            batch_exchange,
            summary_exchange,
            dead_letter_exchange,
        ],
        queues=queues,
    )


class NotificationManager(AppService):
    """Service for handling notifications with batching support"""

    def __init__(self):
        super().__init__()
        self.use_db = True
        self.use_async = False  # Use async RabbitMQ client
        self.use_rabbitmq = create_notification_config()
        self.summary_manager = SummaryManager()
        self.running = True

    @classmethod
    def get_port(cls) -> int:
        return settings.config.notification_service_port

    def get_routing_key(self, event: NotificationEvent) -> str:
        """Get the appropriate routing key for an event"""
        if event.strategy == BatchingStrategy.IMMEDIATE:
            return f"notification.immediate.{event.type.value}"
        elif event.strategy == BatchingStrategy.BACKOFF:
            return f"notification.backoff.{event.type.value}"
        elif event.strategy == BatchingStrategy.HOURLY:
            return f"batch.hourly.{event.type.value}"
        else:  # DAILY
            return f"batch.daily.{event.type.value}"

    @expose
    def queue_notification(self, event: NotificationEvent) -> NotificationResult:
        """Queue a notification - exposed method for other services to call"""
        try:
            routing_key = self.get_routing_key(event)
            message = event.json()

            # Get the appropriate exchange based on strategy
            exchange = None
            if event.strategy in [BatchingStrategy.HOURLY, BatchingStrategy.DAILY]:
                exchange = "batching"
            else:
                exchange = "notifications"

            # Publish to RabbitMQ
            self.run_and_wait(
                self.rabbit.publish_message(
                    routing_key=routing_key,
                    message=message,
                    exchange=next(
                        ex for ex in self.rabbit_config.exchanges if ex.name == exchange
                    ),
                )
            )

            return NotificationResult(
                success=True,
                message=(f"Notification queued with routing key: {routing_key}"),
            )

        except Exception as e:
            logger.error(f"Error queueing notification: {e}")
            return NotificationResult(success=False, message=str(e))

    async def _schedule_next_summary(self, summary_type: str, user_id: str):
        """Schedule the next summary generation using RabbitMQ delayed messages"""
        routing_key = f"summary.{summary_type}"
        message = json.dumps(
            {
                "user_id": user_id,
                "summary_type": summary_type,
                "scheduled_at": datetime.now().isoformat(),
            }
        )

        await self.rabbit.publish_message(
            routing_key=routing_key,
            message=message,
            exchange=next(
                ex for ex in self.rabbit_config.exchanges if ex.name == "summaries"
            ),
        )

    @expose
    async def process_summaries(
        self, summary_type: Optional[str] = None, user_id: Optional[str] = None
    ):
        """
        Process summaries for specified type and user, or all if not specified.
        This is exposed for manual triggering but normally runs on schedule.
        """
        now = datetime.now()
        db = get_db_client()

        summary_configs = {
            "daily": (1, "days"),
            "weekly": (7, "days"),
            "monthly": (1, "months"),
        }

        # If summary_type specified, only process that type
        if summary_type:
            summary_configs = {
                k: v for k, v in summary_configs.items() if k == summary_type
            }

        for summary_type, period_info in summary_configs.items():
            amount, unit = period_info
            start_time = self._get_period_start(now, amount, unit)

            # Calculate end time
            if unit == "months":
                if start_time.month == 12:
                    end_time = datetime(start_time.year + 1, 1, 1)
                else:
                    end_time = datetime(start_time.year, start_time.month + 1, 1)
            else:
                end_time = start_time + timedelta(**{unit: amount})

            # Get users to process
            if user_id:
                # users = [db.get_user(user_id)]
                users = []
            else:
                users = db.get_active_user_ids_in_timerange(
                    start_time.isoformat(), end_time.isoformat()
                )

            for user_id in users:
                await self.summary_manager.generate_summary(
                    summary_type, user_id, start_time, end_time, self
                )
                # Schedule next summary if this wasn't manually triggered
                if not user_id and not summary_type:
                    await self._schedule_next_summary(summary_type, user_id)

    async def _process_summary_trigger(self, message: str):
        """Process a summary trigger message"""
        try:
            data = json.loads(message)
            await self.process_summaries(
                summary_type=data["summary_type"], user_id=data["user_id"]
            )
        except Exception as e:
            logger.error(f"Error processing summary trigger: {e}")

    def _get_period_start(self, now: datetime, amount: int, unit: str) -> datetime:
        """Get the start time for a summary period"""
        if unit == "days":
            if amount == 1:  # Daily summary
                return now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) - timedelta(days=1)
            else:  # Weekly summary
                return now - timedelta(days=now.weekday() + 7)
        else:  # Monthly summary
            if now.month == 1:
                return datetime(now.year - 1, 12, 1)
            else:
                return datetime(now.year, now.month - 1, 1)

    async def _process_notification(self, message: str) -> bool:
        """Process a single notification"""
        try:
            event = NotificationEvent.parse_raw(message)
            # Implementation of actual notification sending would go here
            logger.info(f"Processing notification: {event}")
            return True
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
            return False

    async def _process_batch(self, messages: list[str]) -> bool:
        """Process a batch of notifications"""
        try:
            events = [NotificationEvent.parse_raw(msg) for msg in messages]
            # Implementation of batch notification sending would go here
            logger.info(f"Processing batch of {len(events)} notifications")
            return True
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return False

    def run_service(self):
        logger.info(f"[{self.service_name}] Started notification service")

        # Set up queue consumers
        channel = self.run_and_wait(self.rabbit.get_channel())

        immediate_queue = self.run_and_wait(
            channel.get_queue("immediate_notifications")
        )

        backoff_queue = self.run_and_wait(channel.get_queue("backoff_notifications"))

        hourly_queue = self.run_and_wait(channel.get_queue("hourly_batch"))

        daily_queue = self.run_and_wait(channel.get_queue("daily_batch"))

        # Set up summary queues
        summary_queues = []
        for summary_type in ["daily", "weekly", "monthly"]:
            queue = self.run_and_wait(
                channel.get_queue(f"{summary_type}_summary_trigger")
            )
            summary_queues.append(queue)

        # Initial summary scheduling
        for user_id in get_db_client().get_active_users_ids():
            for summary_type in ["daily", "weekly", "monthly"]:
                self.run_and_wait(self._schedule_next_summary(summary_type, user_id))

        while self.running:
            try:
                # Process immediate notifications
                message = self.run_and_wait(immediate_queue.get())
                if message:
                    success = self.run_and_wait(
                        self._process_notification(message.body.decode())
                    )
                    if success:
                        self.run_and_wait(message.ack())
                    else:
                        self.run_and_wait(message.reject(requeue=False))

                # Process backoff notifications similarly
                message = self.run_and_wait(backoff_queue.get())
                if message:
                    success = self.run_and_wait(
                        self._process_notification(message.body.decode())
                    )
                    if success:
                        self.run_and_wait(message.ack())
                    else:
                        # If failed, will go to DLQ with delay
                        self.run_and_wait(message.reject(requeue=False))

                # Process batch queues
                for queue, batch_size in [
                    (hourly_queue, 50),  # Process up to 50 messages per batch
                    (daily_queue, 100),  # Process up to 100 messages per batch
                ]:
                    messages = []
                    while len(messages) < batch_size:
                        message = self.run_and_wait(queue.get())
                        if not message:
                            break
                        messages.append(message)

                    if messages:
                        success = self.run_and_wait(
                            self._process_batch([msg.body.decode() for msg in messages])
                        )
                        if success:
                            for msg in messages:
                                self.run_and_wait(msg.ack())
                        else:
                            for msg in messages:
                                self.run_and_wait(msg.reject(requeue=True))

                # Process summary triggers
                for queue in summary_queues:
                    message = self.run_and_wait(queue.get())
                    if message:
                        self.run_and_wait(
                            self._process_summary_trigger(message.body.decode())
                        )
                        self.run_and_wait(message.ack())

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in notification service loop: {e}")

    def cleanup(self):
        """Cleanup service resources"""
        self.running = False
        super().cleanup()

    # ------- UTILITIES ------- #


@thread_cached
def get_db_client() -> "DatabaseManager":
    from backend.executor import DatabaseManager

    return get_service_client(DatabaseManager)
