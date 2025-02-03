import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from autogpt_libs.utils.cache import thread_cached
from backend.notifications.models import (
    BatchingStrategy,
    NotificationBatch,
    NotificationEvent,
    NotificationResult,
)

if TYPE_CHECKING:
    from backend.executor import DatabaseManager

from backend.util.service import AppService, expose, get_service_client
from backend.data.redis import get_redis, get_redis_async
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class NotificationManager(AppService):
    """Service for handling notifications with batching support"""

    def __init__(self):
        super().__init__()
        self.use_redis = True
        self.batch_key_prefix = "notification_batch:"
        self.running = True

    @classmethod
    def get_port(cls) -> int:
        return settings.config.notification_service_port

    @expose
    def queue_notification(self, event: NotificationEvent) -> NotificationResult:
        """Queue a notification - exposed method for other services to call"""
        try:
            if event.strategy == BatchingStrategy.IMMEDIATE:
                success = self.run_and_wait(self._process_immediate(event))
                return NotificationResult(
                    success=success,
                    message=(
                        "Immediate notification processed"
                        if success
                        else "Failed to send immediate notification"
                    ),
                )

            success = self.run_and_wait(self._add_to_batch(event))
            return NotificationResult(
                success=success,
                message=(
                    "Notification queued for batch processing"
                    if success
                    else "Failed to queue notification"
                ),
            )

        except Exception as e:
            logger.error(f"Error queueing notification: {e}")
            return NotificationResult(success=False, message=str(e))

    async def _add_to_batch(self, event: NotificationEvent) -> bool:
        """Add an event to its appropriate batch"""
        redis = await get_redis_async()
        batch_key = f"{self.batch_key_prefix}{event.user_id}:{event.strategy}"

        try:
            current_batch = await redis.get(batch_key)
            if current_batch:
                batch = NotificationBatch.parse_raw(current_batch)
                batch.events.append(event)
                batch.last_update = datetime.now()
            else:
                batch = NotificationBatch(
                    user_id=event.user_id, events=[event], strategy=event.strategy
                )

            pipeline = redis.pipeline()
            await pipeline.set(
                batch_key, batch.json(), ex=self._get_batch_expiry(batch.strategy)
            )
            await pipeline.execute()

            # Notify batch processor
            await redis.publish("notification_triggers", batch_key)
            return True

        except Exception as e:
            logger.error(f"Error adding to batch: {e}")
            return False

    async def _process_immediate(self, event: NotificationEvent) -> bool:
        """Process an immediate notification"""
        try:
            # Implementation of actual email sending would go here
            # For now, just log it
            logger.info(f"Sending immediate notification: {event}")
            return True
        except Exception as e:
            logger.error(f"Error processing immediate notification: {e}")
            return False

    def _get_batch_expiry(self, strategy: BatchingStrategy) -> int:
        return {
            BatchingStrategy.HOURLY: 3600,
            BatchingStrategy.DAILY: 86400,
            BatchingStrategy.IMMEDIATE: 300,
        }.get(strategy, 3600)

    def run_service(self):
        """Main service loop - handles batch processing"""
        redis_conn = get_redis()
        pubsub = redis_conn.pubsub()
        pubsub.subscribe("notification_triggers")

        logger.info(f"[{self.service_name}] Started notification service")

        while self.running:
            try:
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message and message["type"] == "message":
                    batch_key = message["data"].decode()
                    self.run_and_wait(self._process_batch(batch_key))

            except Exception as e:
                logger.error(f"Error in notification service loop: {e}")

    async def _process_batch(self, batch_key: str):
        """Process a batch of notifications"""
        redis = await get_redis_async()

        try:
            batch_data = await redis.get(batch_key)
            if not batch_data:
                return

            batch = NotificationBatch.parse_raw(batch_data)
            if not self._should_process_batch(batch):
                return

            # Implementation of batch email sending would go here
            logger.info(f"Processing batch: {batch}")

            await redis.delete(batch_key)

        except Exception as e:
            logger.error(f"Error processing batch {batch_key}: {e}")

    def _should_process_batch(self, batch: NotificationBatch) -> bool:
        age = datetime.now() - batch.last_update
        return (
            len(batch.events) >= 10
            or (batch.strategy == BatchingStrategy.HOURLY and age >= timedelta(hours=1))
            or (batch.strategy == BatchingStrategy.DAILY and age >= timedelta(days=1))
        )

    def cleanup(self):
        """Cleanup service resources"""
        self.running = False
        super().cleanup()

    # ------- UTILITIES ------- #

    # @thread_cached
    # def get_notification_service() -> "NotificationService":
    #     from backend.notifications import NotificationService

    #     return get_service_client(NotificationService)

    @thread_cached
    def get_db_client() -> "DatabaseManager":
        from backend.executor import DatabaseManager

        return get_service_client(DatabaseManager)
