from collections import defaultdict
import logging
from datetime import datetime, timedelta
import time
from typing import TYPE_CHECKING, cast
from autogpt_libs.utils.cache import thread_cached
from backend.notifications.models import (
    BatchingStrategy,
    DailySummaryData,
    MonthlySummaryData,
    NotificationBatch,
    NotificationEvent,
    NotificationResult,
    NotificationType,
    WeeklySummaryData,
    create_notification,
)

if TYPE_CHECKING:
    from backend.executor import DatabaseManager

from backend.notifications.summary import SummaryManager
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
        self.use_db = True
        self.batch_key_prefix = "notification_batch:"
        self.backoff_key_prefix = "notification_backoff:"
        self.summary_manager = SummaryManager()
        self.running = True

    @classmethod
    def get_port(cls) -> int:
        return settings.config.notification_service_port

    async def _get_user_backoff(
        self, user_id: str, notification_type: NotificationType
    ) -> int:
        """Get current backoff attempts for user/notification type"""
        redis = await get_redis_async()
        key = f"{self.backoff_key_prefix}{user_id}:{notification_type.value}"
        attempts = await redis.get(key)
        return int(attempts) if attempts else 0

    async def _increment_backoff(
        self, user_id: str, notification_type: NotificationType
    ) -> int:
        """Increment backoff counter and return new value"""
        redis = await get_redis_async()
        key = f"{self.backoff_key_prefix}{user_id}:{notification_type.value}"
        attempts = await redis.incr(key)

        # Set expiry based on current attempt count
        expiry = min(2**attempts * 300, 86400)  # Start at 5 min, max 24 hours
        await redis.expire(key, expiry)

        return attempts

    async def _reset_backoff(self, user_id: str, notification_type: NotificationType):
        """Reset backoff counter"""
        redis = await get_redis_async()
        key = f"{self.backoff_key_prefix}{user_id}:{notification_type.value}"
        await redis.delete(key)

    @expose
    def queue_notification(self, event: NotificationEvent) -> NotificationResult:
        """Queue a notification - exposed method for other services to call"""
        try:
            if event.strategy == BatchingStrategy.BACKOFF:
                # Handle backoff notifications
                attempts = self.run_and_wait(
                    self._get_user_backoff(event.user_id, event.type)
                )
                if attempts > 0:
                    # Calculate if we should send based on backoff
                    key = f"{self.backoff_key_prefix}{event.user_id}:{event.type.value}"
                    # Check TTL of backoff key
                    redis = self.run_and_wait(get_redis_async())
                    ttl = cast(int, redis.ttl(key)) or 0
                    if ttl > 0:
                        return NotificationResult(
                            success=True,
                            message=f"Notification delayed due to backoff: {ttl}s remaining, {attempts} attempts",
                        )

                success = self.run_and_wait(self._process_immediate(event))
                if success:
                    self.run_and_wait(self._reset_backoff(event.user_id, event.type))
                else:
                    self.run_and_wait(
                        self._increment_backoff(event.user_id, event.type)
                    )
                return NotificationResult(
                    success=success,
                    message="Notification processed with backoff strategy",
                )
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

    def _get_period_start(self, now: datetime, amount: int, unit: str) -> datetime:
        """Get the start time for a summary period

        Args:
            now: Current datetime
            amount: Number of units to look back
            unit: Type of unit ('days' or 'months')

        Returns:
            Start datetime for the period
        """
        if unit == "days":
            if amount == 1:  # Daily summary
                # Start of previous day
                return now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) - timedelta(days=1)
            else:  # Weekly summary
                # Start of previous week (Monday)
                return now - timedelta(days=now.weekday() + 7)
        else:  # Monthly summary
            if now.month == 1:
                # If current month is January, get December of previous year
                return datetime(now.year - 1, 12, 1)
            else:
                # First day of previous month
                return datetime(now.year, now.month - 1, 1)

    async def process_summaries(self):
        redis = await get_redis_async()
        now = datetime.now()
        db = get_db_client()

        for summary_type, period_info in {
            "daily": (1, "days"),
            "weekly": (7, "days"),
            "monthly": (1, "months"),
        }.items():
            if await self.summary_manager.should_generate_summary(summary_type, redis):
                amount, unit = period_info
                start_time = self._get_period_start(now, amount, unit)

                # Calculate end time properly
                if unit == "months":
                    if start_time.month == 12:
                        end_time = datetime(start_time.year + 1, 1, 1)
                    else:
                        end_time = datetime(start_time.year, start_time.month + 1, 1)
                else:
                    end_time = start_time + timedelta(**{unit: amount})

                # Fix: await the async call
                active_users = db.get_active_users_in_timerange(
                    start_time.isoformat(), end_time.isoformat()
                )
                for user in active_users:
                    await self.summary_manager.generate_summary(
                        summary_type, user.id, start_time, end_time, self
                    )
                await redis.set(
                    self.summary_manager.last_check_keys[summary_type], now.isoformat()
                )

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
        redis_conn = get_redis()
        pubsub = redis_conn.pubsub()
        pubsub.subscribe("notification_triggers")

        last_summary_check = datetime.now()
        logger.info(f"[{self.service_name}] Started notification service")

        while self.running:
            try:
                # Process regular notifications
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message and message["type"] == "message":
                    batch_key = message["data"].decode()
                    self.run_and_wait(self._process_batch(batch_key))

                # Check summaries every minute
                now = datetime.now()
                if (now - last_summary_check).total_seconds() >= 60:
                    self.run_and_wait(self.process_summaries())
                    last_summary_check = now

                # Small sleep to prevent CPU spinning
                time.sleep(0.1)

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


@thread_cached
def get_db_client() -> "DatabaseManager":
    from backend.executor import DatabaseManager

    return get_service_client(DatabaseManager)
