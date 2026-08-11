"""The notification service.

Consumes the three notification queues, and owns the two scheduled passes that
the Alert and Briefing families need: flushing matured alerts out of the
debounce window, and assembling briefings for the users whose local morning it
is.
"""

import asyncio
import logging
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Awaitable, Callable

import aio_pika

from backend.data import rabbitmq
from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    BaseEventModel,
    NotificationEventModel,
    NotificationResult,
    get_notif_data_type,
)
from backend.data.user import generate_unsubscribe_link
from backend.notifications import briefing_runner, mailerlite
from backend.notifications.email import EmailSender
from backend.notifications.queue import (
    AUDIENCE_QUEUE,
    OPS_NOTIFICATIONS_QUEUE,
    USER_NOTIFICATIONS_QUEUE,
    create_notification_config,
    queue_notification_async,
)
from backend.notifications.preferences import wants_notification
from backend.util.clients import get_database_manager_async_client
from backend.util.logging import TruncatedLogger
from backend.util.metrics import DiscordChannel, discord_send_alert
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

MAX_CONSUMER_RETRY_ATTEMPTS = 3
CONSUMER_RETRY_BACKOFF_SECONDS = 2
SHUTDOWN_TIMEOUT_SECONDS = 10
CLEANUP_TIMEOUT_SECONDS = SHUTDOWN_TIMEOUT_SECONDS * 2 + 5

__all__ = [
    "NotificationManager",
    "NotificationManagerClient",
    "queue_notification_async",
    "NotificationResult",
]


class NotificationManager(AppService):
    """Renders and sends every email the platform produces."""

    def __init__(self):
        super().__init__()
        self.rabbitmq_config = create_notification_config()
        self.rabbitmq_service: rabbitmq.AsyncRabbitMQ | None = None
        self.running = True
        self.email_sender = EmailSender()
        self._run_service_future: Future[None] | None = None
        self._run_service_task: asyncio.Task[None] | None = None

    @property
    def rabbit(self) -> rabbitmq.AsyncRabbitMQ:
        if not self.rabbitmq_service:
            raise UnhealthyServiceError("RabbitMQ not configured for this service")
        return self.rabbitmq_service

    async def health_check(self) -> str:
        if not self.rabbitmq_service:
            raise UnhealthyServiceError("RabbitMQ not configured for this service")
        if not self.rabbitmq_service.is_ready:
            raise UnhealthyServiceError("RabbitMQ channel is not ready")
        return await super().health_check()

    @classmethod
    def get_port(cls) -> int:
        return settings.config.notification_service_port

    # ── scheduled passes ────────────────────────────────────────────────

    @expose
    async def flush_matured_alerts(self) -> None:
        """Send everything that has sat out the ten-minute debounce window, one
        coalesced email per user."""
        asyncio.create_task(briefing_runner.flush_matured_alerts())

    @expose
    async def send_due_briefings(self) -> None:
        """Assemble and queue briefings for every user whose local ~07:30 this
        hour is."""
        asyncio.create_task(briefing_runner.send_due_briefings())

    @expose
    async def discord_system_alert(
        self, content: str, channel: DiscordChannel = DiscordChannel.PLATFORM
    ):
        try:
            await discord_send_alert(content, channel)
        except Exception as e:
            logger.warning(f"Failed to send Discord system alert: {e}")

    @expose
    async def send_email_or_raise(self, to: str, subject: str, body: str):
        """One-off transactional email (e.g. Better Auth password-reset links
        forwarded by the REST API). Deliberately not wrapped in try/except: a
        delivery failure must reach the RPC caller."""
        await asyncio.to_thread(
            self.email_sender.send_email_or_raise, to, subject, body
        )

    # ── consumers ───────────────────────────────────────────────────────

    async def _process_user_notification(self, message: str) -> bool:
        """A customer-facing notification. Returns False for permanent failures
        (the consumer sends those straight to the DLQ); transient failures
        propagate so the retry-with-backoff loop can recover."""
        event = self._parse_message(message)
        if not event:
            return False

        preference = await get_database_manager_async_client(
            should_retry=False
        ).get_user_notification_preference(event.user_id)
        if not preference.email:
            logger.warning(f"User email not found for user {event.user_id}")
            return False

        verified = await get_database_manager_async_client(
            should_retry=False
        ).get_user_email_verification(event.user_id)
        if not verified or not wants_notification(preference, event.type):
            logger.debug(
                "Skipping %s for user %s: not wanted or unverified",
                event.type,
                event.user_id,
            )
            return True

        await self.email_sender.send_notification(
            notification_type=event.type,
            user_email=preference.email,
            data=event.data,
            unsubscribe_link=generate_unsubscribe_link(event.user_id),
        )
        return True

    async def _process_ops_notification(self, message: str) -> bool:
        """Internal mail to the refunds team. No preference gating: it is not
        opt-in mail, and it carries no unsubscribe."""
        event = self._parse_message(message)
        if not event:
            return False
        recipient = settings.config.refund_notification_email
        await self.email_sender.send_notification(
            notification_type=event.type,
            user_email=recipient,
            data=event.data,
            unsubscribe_link="",
        )
        return True

    async def _process_audience_change(self, message: str) -> bool:
        try:
            event = AudienceEventModel.model_validate_json(message)
        except ValueError as e:
            logger.warning(f"Unparseable audience change (sending to DLQ): {e}")
            return False

        handler = {
            AudienceAction.ENROLL_TOUR: mailerlite.enroll_in_onboarding,
            AudienceAction.ADD_CHANGELOG: mailerlite.add_to_changelog,
            AudienceAction.REMOVE_CHANGELOG: mailerlite.remove_from_changelog,
        }[event.action]
        await handler(event.email)
        return True

    def _parse_message(self, message: str) -> NotificationEventModel | None:
        try:
            event = BaseEventModel.model_validate_json(message)
            return NotificationEventModel[
                get_notif_data_type(event.type)
            ].model_validate_json(message)
        except Exception as e:
            logger.warning(f"Error parsing message due to non matching schema {e}")
            return None

    # ── service lifecycle ───────────────────────────────────────────────

    def run_service(self):
        self._run_service_future = asyncio.run_coroutine_threadsafe(
            self._run_service_with_task_reference(), self.shared_event_loop
        )
        super().run_service()

    async def _run_service_with_task_reference(self) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Notification service did not start in an asyncio task")
        self._run_service_task = task
        try:
            await self._run_service()
        finally:
            if self._run_service_task is task:
                self._run_service_task = None

    @continuous_retry()
    async def _run_service(self):
        logger.info(f"[{self.service_name}] ⏳ Configuring RabbitMQ...")
        self.rabbitmq_service = rabbitmq.AsyncRabbitMQ(self.rabbitmq_config)
        await self.rabbitmq_service.connect()
        logger.info(f"[{self.service_name}] Started notification service")

        channel = await self.rabbit.get_channel()
        await channel.set_qos(prefetch_count=10)

        consumers = {
            USER_NOTIFICATIONS_QUEUE: self._process_user_notification,
            OPS_NOTIFICATIONS_QUEUE: self._process_ops_notification,
            AUDIENCE_QUEUE: self._process_audience_change,
        }
        tasks = [
            asyncio.create_task(
                self._consume_queue(await channel.get_queue(name), handler, name)
            )
            for name, handler in consumers.items()
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Service shutdown requested")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _consume_queue(
        self,
        queue: aio_pika.abc.AbstractQueue,
        process_func: Callable[[str], Awaitable[bool]],
        queue_name: str,
    ):
        logger.info(f"Starting consumer for queue: {queue_name}")
        try:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    if not self.running:
                        break
                    await self._process_message_with_retry(
                        message, process_func, queue_name
                    )
        except asyncio.CancelledError:
            logger.info(f"Consumer for {queue_name} cancelled")
            raise
        except Exception as e:
            logger.exception(f"Fatal error in consumer for {queue_name}: {e}")
            raise

    async def _process_message_with_retry(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
        process_func: Callable[[str], Awaitable[bool]],
        queue_name: str,
    ):
        """Acks on success, rejects (no requeue → DLQ) on permanent failure or
        after exhausting retries.

        ``process_func`` MUST be idempotent: the same body is replayed on each
        attempt, so a partial success (Postmark accepted the email but a later
        write failed) re-runs on retry.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_CONSUMER_RETRY_ATTEMPTS):
            try:
                body = message.body.decode()
                if await process_func(body):
                    await message.ack()
                    return
                logger.warning(
                    f"Message in {queue_name} rejected (process_func returned False)"
                )
                await message.reject(requeue=False)
                return
            except UnicodeDecodeError as e:
                logger.warning(
                    f"Undecodable message in {queue_name}, sending to DLQ: {e}"
                )
                await message.reject(requeue=False)
                return
            except Exception as e:
                last_error = e
                if attempt == MAX_CONSUMER_RETRY_ATTEMPTS - 1:
                    break
                delay = CONSUMER_RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    f"Transient failure on attempt {attempt + 1}/"
                    f"{MAX_CONSUMER_RETRY_ATTEMPTS} in {queue_name}: {e}. "
                    f"Retrying in {delay}s.",
                )
                await asyncio.sleep(delay)
        logger.exception(
            f"Sending message to DLQ from {queue_name} after "
            f"{MAX_CONSUMER_RETRY_ATTEMPTS} attempts. Last error: {last_error}",
            exc_info=last_error,
        )
        await message.reject(requeue=False)

    async def _shutdown_service(self) -> None:
        """Stop consumers completely before closing their RabbitMQ connection."""
        service_future = self._run_service_future
        service_task = self._run_service_task
        while (
            service_task is None
            and service_future is not None
            and not service_future.done()
        ):
            await asyncio.sleep(0)
            service_task = self._run_service_task

        if service_task is not None and service_task is not asyncio.current_task():
            if not service_task.done():
                service_task.cancel()
            _, pending = await asyncio.wait(
                [service_task], timeout=SHUTDOWN_TIMEOUT_SECONDS
            )
            if pending:
                logger.warning(
                    "Notification service task did not finish cancelling within "
                    f"{SHUTDOWN_TIMEOUT_SECONDS}s; continuing shutdown"
                )

        if self.rabbitmq_service is not None:
            logger.info("⏳ Disconnecting RabbitMQ...")
            disconnect_task = asyncio.ensure_future(self.rabbitmq_service.disconnect())
            _, pending = await asyncio.wait(
                [disconnect_task], timeout=SHUTDOWN_TIMEOUT_SECONDS
            )
            if pending:
                disconnect_task.cancel()
                logger.warning(
                    "RabbitMQ disconnect did not complete within "
                    f"{SHUTDOWN_TIMEOUT_SECONDS}s; continuing shutdown"
                )
            elif (
                not disconnect_task.cancelled()
                and (exc := disconnect_task.exception()) is not None
            ):
                logger.warning(f"RabbitMQ disconnect failed during shutdown: {exc}")

    def cleanup(self):
        self.running = False
        try:
            shutdown = self._shutdown_service()
            if self.shared_event_loop.is_closed():
                shutdown.close()
                logger.warning(
                    "Event loop is already closed; "
                    "skipping notification service shutdown"
                )
            elif self.shared_event_loop.is_running():
                try:
                    self.run_and_wait(shutdown, timeout=CLEANUP_TIMEOUT_SECONDS)
                except FutureTimeoutError:
                    logger.warning(
                        "Notification service shutdown did not run within "
                        f"{CLEANUP_TIMEOUT_SECONDS}s; continuing cleanup"
                    )
                except RuntimeError as e:
                    shutdown.close()
                    logger.warning(f"Could not run notification service shutdown: {e}")
            else:
                self.shared_event_loop.run_until_complete(shutdown)
        finally:
            self._run_service_future = None
            self._run_service_task = None
            super().cleanup()


class NotificationManagerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return NotificationManager

    flush_matured_alerts = endpoint_to_sync(NotificationManager.flush_matured_alerts)
    send_due_briefings = endpoint_to_sync(NotificationManager.send_due_briefings)
    discord_system_alert = endpoint_to_sync(NotificationManager.discord_system_alert)
    send_email_or_raise = endpoint_to_sync(NotificationManager.send_email_or_raise)
