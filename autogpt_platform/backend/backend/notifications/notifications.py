import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

import aio_pika
from prisma.enums import NotificationType

from backend.data import rabbitmq
from backend.data.notifications import (
    BaseEventModel,
    BaseSummaryData,
    BaseSummaryParams,
    DailySummaryData,
    DailySummaryParams,
    NotificationEventModel,
    NotificationResult,
    NotificationTypeOverride,
    QueueType,
    SummaryParamsEventModel,
    WeeklySummaryData,
    WeeklySummaryParams,
    get_batch_delay,
    get_notif_data_type,
    get_summary_params_type,
)
from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.data.user import generate_unsubscribe_link
from backend.notifications.email import EmailSender
from backend.util.clients import get_database_manager_async_client
from backend.util.logging import TruncatedLogger
from backend.util.metrics import discord_send_alert
from backend.util.retry import continuous_retry
from backend.util.service import (
    AppService,
    AppServiceClient,
    UnhealthyServiceError,
    endpoint_to_sync,
    expose,
)
from backend.util.settings import Settings

logger = TruncatedLogger(logging.getLogger(__name__), "[NotificationManager]")
settings = Settings()


NOTIFICATION_EXCHANGE = Exchange(name="notifications", type=ExchangeType.TOPIC)
DEAD_LETTER_EXCHANGE = Exchange(name="dead_letter", type=ExchangeType.TOPIC)
EXCHANGES = [NOTIFICATION_EXCHANGE, DEAD_LETTER_EXCHANGE]


def create_notification_config() -> RabbitMQConfig:
    """Create RabbitMQ configuration for notifications"""

    queues = [
        # Main notification queues
        Queue(
            name="immediate_notifications",
            exchange=NOTIFICATION_EXCHANGE,
            routing_key="notification.immediate.#",
            arguments={
                "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
                "x-dead-letter-routing-key": "failed.immediate",
            },
        ),
        Queue(
            name="admin_notifications",
            exchange=NOTIFICATION_EXCHANGE,
            routing_key="notification.admin.#",
            arguments={
                "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
                "x-dead-letter-routing-key": "failed.admin",
            },
        ),
        # Summary notification queues
        Queue(
            name="summary_notifications",
            exchange=NOTIFICATION_EXCHANGE,
            routing_key="notification.summary.#",
            arguments={
                "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
                "x-dead-letter-routing-key": "failed.summary",
            },
        ),
        # Batch Queue
        Queue(
            name="batch_notifications",
            exchange=NOTIFICATION_EXCHANGE,
            routing_key="notification.batch.#",
            arguments={
                "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
                "x-dead-letter-routing-key": "failed.batch",
            },
        ),
        # Failed notifications queue
        Queue(
            name="failed_notifications",
            exchange=DEAD_LETTER_EXCHANGE,
            routing_key="failed.#",
        ),
    ]

    return RabbitMQConfig(
        exchanges=EXCHANGES,
        queues=queues,
    )


def get_routing_key(event_type: NotificationType) -> str:
    strategy = NotificationTypeOverride(event_type).strategy
    """Get the appropriate routing key for an event"""
    if strategy == QueueType.IMMEDIATE:
        return f"notification.immediate.{event_type.value}"
    elif strategy == QueueType.BACKOFF:
        return f"notification.backoff.{event_type.value}"
    elif strategy == QueueType.ADMIN:
        return f"notification.admin.{event_type.value}"
    elif strategy == QueueType.BATCH:
        return f"notification.batch.{event_type.value}"
    elif strategy == QueueType.SUMMARY:
        return f"notification.summary.{event_type.value}"
    return f"notification.{event_type.value}"


def queue_notification(event: NotificationEventModel) -> NotificationResult:
    """Queue a notification - exposed method for other services to call"""
    try:
        logger.debug(f"Received Request to queue {event=}")

        exchange = "notifications"
        routing_key = get_routing_key(event.type)

        from backend.util.clients import get_notification_queue

        queue = get_notification_queue()
        queue.publish_message(
            routing_key=routing_key,
            message=event.model_dump_json(),
            exchange=next(ex for ex in EXCHANGES if ex.name == exchange),
        )

        return NotificationResult(
            success=True,
            message=f"Notification queued with routing key: {routing_key}",
        )

    except Exception as e:
        logger.exception(f"Error queueing notification: {e}")
        return NotificationResult(success=False, message=str(e))


async def queue_notification_async(event: NotificationEventModel) -> NotificationResult:
    """Queue a notification - exposed method for other services to call"""
    try:
        logger.debug(f"Received Request to queue {event=}")

        exchange = "notifications"
        routing_key = get_routing_key(event.type)

        from backend.util.clients import get_async_notification_queue

        queue = await get_async_notification_queue()
        await queue.publish_message(
            routing_key=routing_key,
            message=event.model_dump_json(),
            exchange=next(ex for ex in EXCHANGES if ex.name == exchange),
        )

        return NotificationResult(
            success=True,
            message=f"Notification queued with routing key: {routing_key}",
        )

    except Exception as e:
        logger.exception(f"Error queueing notification: {e}")
        return NotificationResult(success=False, message=str(e))


class NotificationManager(AppService):
    """Service for handling notifications with batching support"""

    def __init__(self):
        super().__init__()
        self.rabbitmq_config = create_notification_config()
        self.running = True
        self.email_sender = EmailSender()

    @property
    def rabbit(self) -> rabbitmq.AsyncRabbitMQ:
        """Access the RabbitMQ service. Will raise if not configured."""
        if not hasattr(self, "rabbitmq_service") or not self.rabbitmq_service:
            raise UnhealthyServiceError("RabbitMQ not configured for this service")
        return self.rabbitmq_service

    @property
    def rabbit_config(self) -> rabbitmq.RabbitMQConfig:
        """Access the RabbitMQ config. Will raise if not configured."""
        if not self.rabbitmq_config:
            raise UnhealthyServiceError("RabbitMQ not configured for this service")
        return self.rabbitmq_config

    def health_check(self) -> str:
        # Service is unhealthy if RabbitMQ is not ready
        if not hasattr(self, "rabbitmq_service") or not self.rabbitmq_service:
            raise UnhealthyServiceError("RabbitMQ not configured for this service")
        if not self.rabbitmq_service.is_ready:
            raise UnhealthyServiceError("RabbitMQ channel is not ready")
        return super().health_check()

    @classmethod
    def get_port(cls) -> int:
        return settings.config.notification_service_port

    @expose
    async def queue_weekly_summary(self):
        # Use the existing event loop instead of creating a new one with asyncio.run()
        asyncio.create_task(self._queue_weekly_summary())

    async def _queue_weekly_summary(self):
        """Process weekly summary for specified notification types"""
        try:
            logger.info("Processing weekly summary queuing operation")
            processed_count = 0
            current_time = datetime.now(tz=timezone.utc)
            start_time = current_time - timedelta(days=7)
            users = await get_database_manager_async_client().get_active_user_ids_in_timerange(
                end_time=current_time.isoformat(),
                start_time=start_time.isoformat(),
            )
            for user in users:
                await self._queue_scheduled_notification(
                    SummaryParamsEventModel(
                        user_id=user,
                        type=NotificationType.WEEKLY_SUMMARY,
                        data=WeeklySummaryParams(
                            start_date=start_time,
                            end_date=current_time,
                        ),
                    ),
                )
                processed_count += 1

            logger.info(f"Processed {processed_count} weekly summaries into queue")

        except Exception as e:
            logger.exception(f"Error processing weekly summary: {e}")

    @expose
    async def process_existing_batches(
        self, notification_types: list[NotificationType]
    ):
        # Use the existing event loop instead of creating a new process
        asyncio.create_task(self._process_existing_batches(notification_types))

    async def _process_existing_batches(
        self, notification_types: list[NotificationType]
    ):
        """Process existing batches for specified notification types"""
        try:
            processed_count = 0
            current_time = datetime.now(tz=timezone.utc)

            for notification_type in notification_types:
                # Get all batches for this notification type
                batches = (
                    await get_database_manager_async_client().get_all_batches_by_type(
                        notification_type
                    )
                )

                for batch in batches:
                    # Check if batch has aged out
                    oldest_message = await get_database_manager_async_client().get_user_notification_oldest_message_in_batch(
                        batch.user_id, notification_type
                    )

                    if not oldest_message:
                        # this should never happen
                        logger.error(
                            f"Batch for user {batch.user_id} and type {notification_type} has no oldest message whichshould never happen!!!!!!!!!!!!!!!!"
                        )
                        continue

                    max_delay = get_batch_delay(notification_type)

                    # If batch has aged out, process it
                    if oldest_message.created_at + max_delay < current_time:
                        recipient_email = await get_database_manager_async_client().get_user_email_by_id(
                            batch.user_id
                        )

                        if not recipient_email:
                            logger.error(
                                f"User email not found for user {batch.user_id}"
                            )
                            continue

                        should_send = await self._should_email_user_based_on_preference(
                            batch.user_id, notification_type
                        )

                        if not should_send:
                            logger.debug(
                                f"User {batch.user_id} does not want to receive {notification_type} notifications"
                            )
                            # Clear the batch
                            await get_database_manager_async_client().empty_user_notification_batch(
                                batch.user_id, notification_type
                            )
                            continue

                        batch_data = await get_database_manager_async_client().get_user_notification_batch(
                            batch.user_id, notification_type
                        )

                        if not batch_data or not batch_data.notifications:
                            logger.error(
                                f"Batch data not found for user {batch.user_id}"
                            )
                            # Clear the batch
                            await get_database_manager_async_client().empty_user_notification_batch(
                                batch.user_id, notification_type
                            )
                            continue

                        unsub_link = generate_unsubscribe_link(batch.user_id)
                        events = []
                        for db_event in batch_data.notifications:
                            try:
                                events.append(
                                    NotificationEventModel[
                                        get_notif_data_type(db_event.type)
                                    ].model_validate(
                                        {
                                            "user_id": batch.user_id,
                                            "type": db_event.type,
                                            "data": db_event.data,
                                            "created_at": db_event.created_at,
                                        }
                                    )
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error parsing notification event: {e=}, {db_event=}"
                                )
                                continue
                        logger.info(f"{events=}")

                        self.email_sender.send_templated(
                            notification=notification_type,
                            user_email=recipient_email,
                            data=events,
                            user_unsub_link=unsub_link,
                        )

                        # Clear the batch
                        await get_database_manager_async_client().empty_user_notification_batch(
                            batch.user_id, notification_type
                        )

                        processed_count += 1

            logger.info(f"Processed {processed_count} aged batches")
            return {
                "success": True,
                "processed_count": processed_count,
                "notification_types": [nt.value for nt in notification_types],
                "timestamp": current_time.isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error processing batches: {e}")
            return {
                "success": False,
                "error": str(e),
                "notification_types": [nt.value for nt in notification_types],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

    @expose
    async def discord_system_alert(self, content: str):
        await discord_send_alert(content)

    async def _queue_scheduled_notification(self, event: SummaryParamsEventModel):
        """Queue a scheduled notification - exposed method for other services to call"""
        try:
            logger.debug(f"Received Request to queue scheduled notification {event=}")

            exchange = "notifications"
            routing_key = get_routing_key(event.type)

            # Publish to RabbitMQ
            await self.rabbit.publish_message(
                routing_key=routing_key,
                message=event.model_dump_json(),
                exchange=next(ex for ex in EXCHANGES if ex.name == exchange),
            )

        except Exception as e:
            logger.exception(f"Error queueing notification: {e}")

    async def _should_email_user_based_on_preference(
        self, user_id: str, event_type: NotificationType
    ) -> bool:
        """Check if a user wants to receive a notification based on their preferences and email verification status"""
        validated_email = (
            await get_database_manager_async_client().get_user_email_verification(
                user_id
            )
        )
        preference = (
            await get_database_manager_async_client().get_user_notification_preference(
                user_id
            )
        ).preferences.get(event_type, True)
        # only if both are true, should we email this person
        return validated_email and preference

    def _gather_summary_data(
        self, user_id: str, event_type: NotificationType, params: BaseSummaryParams
    ) -> BaseSummaryData:
        """Gathers the data to build a summary notification"""

        logger.info(
            f"Gathering summary data for {user_id} and {event_type} wiht {params=}"
        )

        # total_credits_used = self.run_and_wait(
        #     get_total_credits_used(user_id, start_time, end_time)
        # )

        # total_executions = self.run_and_wait(
        #     get_total_executions(user_id, start_time, end_time)
        # )

        # most_used_agent = self.run_and_wait(
        #     get_most_used_agent(user_id, start_time, end_time)
        # )

        # execution_times = self.run_and_wait(
        #     get_execution_time(user_id, start_time, end_time)
        # )

        # runs = self.run_and_wait(
        #     get_runs(user_id, start_time, end_time)
        # )
        total_credits_used = 3.0
        total_executions = 2
        most_used_agent = {"name": "Some"}
        execution_times = [1, 2, 3]
        runs = [{"status": "COMPLETED"}, {"status": "FAILED"}]

        successful_runs = len([run for run in runs if run["status"] == "COMPLETED"])
        failed_runs = len([run for run in runs if run["status"] != "COMPLETED"])
        average_execution_time = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )
        # cost_breakdown = self.run_and_wait(
        #     get_cost_breakdown(user_id, start_time, end_time)
        # )

        cost_breakdown = {
            "agent1": 1.0,
            "agent2": 2.0,
        }

        if event_type == NotificationType.DAILY_SUMMARY and isinstance(
            params, DailySummaryParams
        ):
            return DailySummaryData(
                total_credits_used=total_credits_used,
                total_executions=total_executions,
                most_used_agent=most_used_agent["name"],
                total_execution_time=sum(execution_times),
                successful_runs=successful_runs,
                failed_runs=failed_runs,
                average_execution_time=average_execution_time,
                cost_breakdown=cost_breakdown,
                date=params.date,
            )
        elif event_type == NotificationType.WEEKLY_SUMMARY and isinstance(
            params, WeeklySummaryParams
        ):
            return WeeklySummaryData(
                total_credits_used=total_credits_used,
                total_executions=total_executions,
                most_used_agent=most_used_agent["name"],
                total_execution_time=sum(execution_times),
                successful_runs=successful_runs,
                failed_runs=failed_runs,
                average_execution_time=average_execution_time,
                cost_breakdown=cost_breakdown,
                start_date=params.start_date,
                end_date=params.end_date,
            )
        else:
            raise ValueError("Invalid event type or params")

    async def _should_batch(
        self, user_id: str, event_type: NotificationType, event: NotificationEventModel
    ) -> bool:

        await get_database_manager_async_client().create_or_add_to_user_notification_batch(
            user_id, event_type, event
        )

        oldest_message = await get_database_manager_async_client().get_user_notification_oldest_message_in_batch(
            user_id, event_type
        )
        if not oldest_message:
            logger.error(
                f"Batch for user {user_id} and type {event_type} has no oldest message whichshould never happen!!!!!!!!!!!!!!!!"
            )
            return False
        oldest_age = oldest_message.created_at

        max_delay = get_batch_delay(event_type)

        if oldest_age + max_delay < datetime.now(tz=timezone.utc):
            logger.info(f"Batch for user {user_id} and type {event_type} is old enough")
            return True
        logger.info(
            f"Batch for user {user_id} and type {event_type} is not old enough: {oldest_age + max_delay} < {datetime.now(tz=timezone.utc)} max_delay={max_delay}"
        )
        return False

    def _parse_message(self, message: str) -> NotificationEventModel | None:
        try:
            event = BaseEventModel.model_validate_json(message)
            return NotificationEventModel[
                get_notif_data_type(event.type)
            ].model_validate_json(message)
        except Exception as e:
            logger.error(f"Error parsing message due to non matching schema {e}")
            return None

    async def _process_admin_message(self, message: str) -> bool:
        """Process a single notification, sending to an admin, returning whether to put into the failed queue"""
        try:
            event = self._parse_message(message)
            if not event:
                return False
            logger.debug(f"Processing notification for admin: {event}")
            recipient_email = settings.config.refund_notification_email
            self.email_sender.send_templated(event.type, recipient_email, event)
            return True
        except Exception as e:
            logger.exception(f"Error processing notification for admin queue: {e}")
            return False

    async def _process_immediate(self, message: str) -> bool:
        """Process a single notification immediately, returning whether to put into the failed queue"""
        try:
            event = self._parse_message(message)
            if not event:
                return False
            logger.debug(f"Processing immediate notification: {event}")

            recipient_email = (
                await get_database_manager_async_client().get_user_email_by_id(
                    event.user_id
                )
            )
            if not recipient_email:
                logger.error(f"User email not found for user {event.user_id}")
                return False

            should_send = await self._should_email_user_based_on_preference(
                event.user_id, event.type
            )
            if not should_send:
                logger.debug(
                    f"User {event.user_id} does not want to receive {event.type} notifications"
                )
                return True

            unsub_link = generate_unsubscribe_link(event.user_id)

            self.email_sender.send_templated(
                notification=event.type,
                user_email=recipient_email,
                data=event,
                user_unsub_link=unsub_link,
            )
            return True
        except Exception as e:
            logger.exception(f"Error processing notification for immediate queue: {e}")
            return False

    async def _process_batch(self, message: str) -> bool:
        """Process a single notification with a batching strategy, returning whether to put into the failed queue"""
        try:
            event = self._parse_message(message)
            if not event:
                return False
            logger.info(f"Processing batch notification: {event}")

            recipient_email = (
                await get_database_manager_async_client().get_user_email_by_id(
                    event.user_id
                )
            )
            if not recipient_email:
                logger.error(f"User email not found for user {event.user_id}")
                return False

            should_send = await self._should_email_user_based_on_preference(
                event.user_id, event.type
            )
            if not should_send:
                logger.info(
                    f"User {event.user_id} does not want to receive {event.type} notifications"
                )
                return True

            should_send = await self._should_batch(event.user_id, event.type, event)

            if not should_send:
                logger.info("Batch not old enough to send")
                return False
            batch = (
                await get_database_manager_async_client().get_user_notification_batch(
                    event.user_id, event.type
                )
            )
            if not batch or not batch.notifications:
                logger.error(f"Batch not found for user {event.user_id}")
                return False
            unsub_link = generate_unsubscribe_link(event.user_id)

            batch_messages = [
                NotificationEventModel[
                    get_notif_data_type(db_event.type)
                ].model_validate(
                    {
                        "user_id": event.user_id,
                        "type": db_event.type,
                        "data": db_event.data,
                        "created_at": db_event.created_at,
                    }
                )
                for db_event in batch.notifications
            ]

            # Split batch into chunks to avoid exceeding email size limits
            # Start with a reasonable chunk size and adjust dynamically
            MAX_EMAIL_SIZE = 4_500_000  # 4.5MB to leave buffer under 5MB limit
            chunk_size = 100  # Initial chunk size
            successfully_sent_count = 0
            failed_indices = []

            i = 0
            while i < len(batch_messages):
                # Try progressively smaller chunks if needed
                chunk_sent = False
                for attempt_size in [chunk_size, 50, 25, 10, 5, 1]:
                    chunk = batch_messages[i : i + attempt_size]

                    try:
                        # Try to render the email to check its size
                        template = self.email_sender._get_template(event.type)
                        _, test_message = self.email_sender.formatter.format_email(
                            base_template=template.base_template,
                            subject_template=template.subject_template,
                            content_template=template.body_template,
                            data={"notifications": chunk},
                            unsubscribe_link=f"{self.email_sender.formatter.env.globals.get('base_url', '')}/profile/settings",
                        )

                        if len(test_message) < MAX_EMAIL_SIZE:
                            # Size is acceptable, send the email
                            logger.info(
                                f"Sending email with {len(chunk)} notifications "
                                f"(size: {len(test_message):,} chars)"
                            )

                            self.email_sender.send_templated(
                                notification=event.type,
                                user_email=recipient_email,
                                data=chunk,
                                user_unsub_link=unsub_link,
                            )

                            # Track successful sends
                            successfully_sent_count += len(chunk)

                            # Update chunk_size for next iteration based on success
                            if (
                                attempt_size == chunk_size
                                and len(test_message) < MAX_EMAIL_SIZE * 0.7
                            ):
                                # If we're well under limit, try larger chunks next time
                                chunk_size = min(chunk_size + 10, 100)
                            elif len(test_message) > MAX_EMAIL_SIZE * 0.9:
                                # If we're close to limit, use smaller chunks
                                chunk_size = max(attempt_size - 10, 1)

                            i += len(chunk)
                            chunk_sent = True
                            break
                    except Exception as e:
                        if attempt_size == 1:
                            # Even single notification is too large
                            logger.error(
                                f"Single notification too large to send: {e}. "
                                f"Skipping notification at index {i}"
                            )
                            failed_indices.append(i)
                            i += 1
                            chunk_sent = True
                            break
                        # Try smaller chunk
                        continue

                if not chunk_sent:
                    # Should not reach here due to single notification handling
                    logger.error(f"Failed to send notifications starting at index {i}")
                    failed_indices.append(i)
                    i += 1

            # Only empty the batch if ALL notifications were sent successfully
            if successfully_sent_count == len(batch_messages):
                logger.info(
                    f"Successfully sent all {successfully_sent_count} notifications, clearing batch"
                )
                await get_database_manager_async_client().empty_user_notification_batch(
                    event.user_id, event.type
                )
            else:
                logger.warning(
                    f"Only sent {successfully_sent_count} of {len(batch_messages)} notifications. "
                    f"Failed indices: {failed_indices}. Batch will be retained for retry."
                )
            return True
        except Exception as e:
            logger.exception(f"Error processing notification for batch queue: {e}")
            return False

    async def _process_summary(self, message: str) -> bool:
        """Process a single notification with a summary strategy, returning whether to put into the failed queue"""
        try:
            logger.info(f"Processing summary notification: {message}")
            event = BaseEventModel.model_validate_json(message)
            model = SummaryParamsEventModel[
                get_summary_params_type(event.type)
            ].model_validate_json(message)

            logger.info(f"Processing summary notification: {model}")

            recipient_email = (
                await get_database_manager_async_client().get_user_email_by_id(
                    event.user_id
                )
            )
            if not recipient_email:
                logger.error(f"User email not found for user {event.user_id}")
                return False
            should_send = await self._should_email_user_based_on_preference(
                event.user_id, event.type
            )
            if not should_send:
                logger.info(
                    f"User {event.user_id} does not want to receive {event.type} notifications"
                )
                return True

            summary_data = self._gather_summary_data(
                event.user_id, event.type, model.data
            )

            unsub_link = generate_unsubscribe_link(event.user_id)

            data = NotificationEventModel(
                user_id=event.user_id,
                type=event.type,
                data=summary_data,
            )

            self.email_sender.send_templated(
                notification=event.type,
                user_email=recipient_email,
                data=data,
                user_unsub_link=unsub_link,
            )
            return True
        except Exception as e:
            logger.exception(f"Error processing notification for summary queue: {e}")
            return False

    async def _consume_queue(
        self,
        queue: aio_pika.abc.AbstractQueue,
        process_func: Callable[[str], Awaitable[bool]],
        queue_name: str,
    ):
        """Continuously consume messages from a queue using async iteration"""
        logger.info(f"Starting consumer for queue: {queue_name}")

        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    if not self.running:
                        break

                    try:
                        async with message.process():
                            result = await process_func(message.body.decode())
                            if not result:
                                # Message will be rejected when exiting context without exception
                                raise aio_pika.exceptions.MessageProcessError(
                                    "Processing failed"
                                )
                    except aio_pika.exceptions.MessageProcessError:
                        # Let message.process() handle the rejection
                        pass
                    except Exception as e:
                        logger.error(f"Error processing message in {queue_name}: {e}")
                        # Let message.process() handle the rejection
                        raise
        except asyncio.CancelledError:
            logger.info(f"Consumer for {queue_name} cancelled")
            raise
        except Exception as e:
            logger.exception(f"Fatal error in consumer for {queue_name}: {e}")
            raise

    @continuous_retry()
    def run_service(self):
        self.run_and_wait(self._run_service())

    async def _run_service(self):
        logger.info(f"[{self.service_name}] ⏳ Configuring RabbitMQ...")
        self.rabbitmq_service = rabbitmq.AsyncRabbitMQ(self.rabbitmq_config)
        await self.rabbitmq_service.connect()

        logger.info(f"[{self.service_name}] Started notification service")

        # Set up queue consumers with QoS settings
        channel = await self.rabbit.get_channel()

        # Set prefetch to prevent overwhelming the service
        await channel.set_qos(prefetch_count=10)

        immediate_queue = await channel.get_queue("immediate_notifications")
        batch_queue = await channel.get_queue("batch_notifications")
        admin_queue = await channel.get_queue("admin_notifications")
        summary_queue = await channel.get_queue("summary_notifications")

        # Create consumer tasks for each queue - running in parallel
        consumer_tasks = [
            asyncio.create_task(
                self._consume_queue(
                    queue=immediate_queue,
                    process_func=self._process_immediate,
                    queue_name="immediate_notifications",
                )
            ),
            asyncio.create_task(
                self._consume_queue(
                    queue=admin_queue,
                    process_func=self._process_admin_message,
                    queue_name="admin_notifications",
                )
            ),
            asyncio.create_task(
                self._consume_queue(
                    queue=batch_queue,
                    process_func=self._process_batch,
                    queue_name="batch_notifications",
                )
            ),
            asyncio.create_task(
                self._consume_queue(
                    queue=summary_queue,
                    process_func=self._process_summary,
                    queue_name="summary_notifications",
                )
            ),
        ]

        try:
            # Run all consumers concurrently
            await asyncio.gather(*consumer_tasks)
        except asyncio.CancelledError:
            logger.info("Service shutdown requested")
            # Cancel all consumer tasks
            for task in consumer_tasks:
                task.cancel()
            # Wait for all tasks to complete cancellation
            await asyncio.gather(*consumer_tasks, return_exceptions=True)
            raise

    def cleanup(self):
        """Cleanup service resources"""
        self.running = False
        super().cleanup()
        logger.info(f"[{self.service_name}] ⏳ Disconnecting RabbitMQ...")
        self.run_and_wait(self.rabbitmq_service.disconnect())


class NotificationManagerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return NotificationManager

    process_existing_batches = endpoint_to_sync(
        NotificationManager.process_existing_batches
    )
    queue_weekly_summary = endpoint_to_sync(NotificationManager.queue_weekly_summary)
    discord_system_alert = endpoint_to_sync(NotificationManager.discord_system_alert)
