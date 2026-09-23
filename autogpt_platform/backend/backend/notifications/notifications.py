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
from datetime import date, datetime, timezone
from enum import Enum, auto
from typing import Awaitable, Callable, Coroutine

import aio_pika
from aio_pika.exceptions import ChannelInvalidStateError
from postmarker.exceptions import ClientError
from prisma.enums import NotificationType

from backend.data import rabbitmq
from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    BaseEventModel,
    NotificationEventModel,
    NotificationResult,
    PassWorkEvent,
    TrialUpdateData,
    get_notif_data_type,
)
from backend.data.user import (
    FOOTER_CHOICES,
    generate_preference_link,
    generate_unsubscribe_link,
)
from backend.notifications import briefing_runner, mailerlite
from backend.notifications.dedupe import claim_daily_send
from backend.notifications.email import EmailSender
from backend.notifications.preferences import SERVICE_MESSAGES, wants_notification
from backend.notifications.queue import (
    AUDIENCE_QUEUE,
    OPS_NOTIFICATIONS_QUEUE,
    PASS_WORK_QUEUE,
    USER_NOTIFICATIONS_QUEUE,
    create_notification_config,
    queue_notification_async,
)
from backend.notifications.trial import trial_notice_disposition
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


def _utc_today() -> date:
    """The cap resets at UTC midnight, matching the Alert engine's own daily
    cap. Both are documented as a follow-up to move to the user's local day."""
    return datetime.now(tz=timezone.utc).date()


def _is_channel_loss(error: BaseException) -> bool:
    """Every leaf is the queue iterator failing to nack on a dead channel.

    `split` walks nested groups too: the TaskGroup wraps whatever the
    iterator's exit raised, so the nack failures can sit one level down.
    """
    if not isinstance(error, BaseExceptionGroup):
        return False
    _, rest = error.split(ChannelInvalidStateError)
    return rest is None


def _is_permanent_delivery_failure(error: ClientError) -> bool:
    return error.error_code in PERMANENT_POSTMARK_ERROR_CODES


MAX_CONSUMER_RETRY_ATTEMPTS = 3
CONSUMER_RETRY_BACKOFF_SECONDS = 2
# Messages a consumer works on at once, and the broker prefetch to match. The
# work is I/O bound (DatabaseManager RPCs, Postmark); one at a time left the
# other prefetched messages waiting in memory, and a large fan-out from one
# scheduled pass drained at a fraction of the rate the pass published it.
CONSUMER_CONCURRENCY = 10
# Hard ceiling on one message's processing. RabbitMQ closes the channel when a
# delivered message goes unacknowledged for its consumer timeout (30 minutes
# by default), and every other in-flight ack on that channel then fails too;
# a bounded wait turns one hung call into one retried message instead.
MESSAGE_PROCESSING_TIMEOUT_SECONDS = 300
# How long in-flight handlers get to settle before they are cancelled, on
# both of the ways a consumer comes down: a shutdown, and a handler that
# cannot settle its own message. A handler cancelled between its send and its
# ack leaves an email delivered and the message unacked, which the broker then
# redelivers.
#
# The guarantee is narrower than that reads, so state it plainly: a handler
# that finishes inside the grace is acked, and a handler still running at the
# end of it is cancelled and redelivered exactly as it was before. The second
# case is not hypothetical — a briefing pass does DatabaseManager RPCs, then
# renders, then calls Postmark, whose own client timeout is 30s — so this
# narrows the double-send window rather than closing it.
#
# It is not simply raised to cover that: the grace has to fit inside
# SHUTDOWN_TIMEOUT_SECONDS with room for the cancel itself, and that sets
# CLEANUP_TIMEOUT_SECONDS, the whole budget the process gets before its
# supervisor stops waiting and kills it. Covering a 30s send means pushing all
# three past a typical termination grace, trading a rare redelivered email for
# a reliable hard kill mid-send.
HANDLER_SHUTDOWN_GRACE_SECONDS = 5
# Postmark error codes that will never succeed on retry: 300 is a malformed
# request (bad address), 406 an inactive recipient (hard bounce, spam
# complaint or suppression). Retrying only delays the dead-letter by seconds.
PERMANENT_POSTMARK_ERROR_CODES = frozenset({300, 406})
SHUTDOWN_TIMEOUT_SECONDS = 10
CLEANUP_TIMEOUT_SECONDS = SHUTDOWN_TIMEOUT_SECONDS * 2 + 5


class Ordering(Enum):
    """Whether one queue's messages may be worked out of order.

    Every consumer declares this where it is registered in `_run_service`,
    because that is where a handler is added and the question has to be
    answered: working the prefetch concurrently drops FIFO within the queue,
    which is only safe where two of its messages commute.

    COMMUTATIVE means they do — a second briefing or a second ops mail is its
    own unit of work — and the queue runs at CONSUMER_CONCURRENCY.
    AS_PUBLISHED means they do not, and the queue is worked one message at a
    time however many the broker prefetches.
    """

    COMMUTATIVE = auto()
    AS_PUBLISHED = auto()

    @property
    def concurrency(self) -> int:
        return 1 if self is Ordering.AS_PUBLISHED else CONSUMER_CONCURRENCY


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
        # In-flight scheduled passes, keyed by name, so a slow tick cannot
        # overlap the next one and double-send.
        self._passes: dict[str, asyncio.Task[None]] = {}

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
        self._spawn_pass("flush_matured_alerts", briefing_runner.flush_matured_alerts)

    @expose
    async def send_due_briefings(self) -> None:
        """Assemble and queue briefings for every user whose local ~07:30 this
        hour is."""
        self._spawn_pass("send_due_briefings", briefing_runner.send_due_briefings)

    def _spawn_pass(
        self, name: str, work: Callable[[], Coroutine[None, None, None]]
    ) -> None:
        """Run a scheduled pass in the background, at most one at a time.

        The scheduler fires these on a fixed interval and this RPC returns as
        soon as the task is spawned, so nothing else stops a slow pass from
        overlapping the next tick. Both passes queue their email *before*
        marking the rows that suppress a resend, so two concurrent runs read
        the same PENDING conditions and send the same alert twice.

        The task is also held in a dict rather than left to float: a bare
        `create_task` reference can be garbage-collected mid-flight.
        """
        existing = self._passes.get(name)
        if existing and not existing.done():
            logger.warning(
                f"{name} is still running from the previous tick; skipping this one"
            )
            return

        task = asyncio.create_task(work(), name=name)
        self._passes[name] = task
        task.add_done_callback(self._clear_pass)

    def _clear_pass(self, task: asyncio.Task) -> None:
        """Drop a finished pass, but only if it is still the registered one.

        Done callbacks are dispatched via `call_soon`, so a task that finished
        just before the next tick can have its callback run *after* the
        successor is registered. Popping by name alone would clear the guard
        out from under a pass that is still running.
        """
        name = task.get_name()
        if self._passes.get(name) is task:
            del self._passes[name]
        if task.cancelled():
            # Shutdown cancels these; `task.exception()` would re-raise the
            # CancelledError inside the done-callback.
            return
        if (exc := task.exception()) is not None:
            logger.error(f"Scheduled pass {name} failed: {exc}", exc_info=exc)

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

        if event.type == NotificationType.TRIAL_UPDATE:
            data = TrialUpdateData.model_validate(event.data.model_dump())
            disposition = await trial_notice_disposition(event.user_id, data)
            if disposition == "obsolete":
                return True
            if disposition == "suppressed":
                # Keep the claim: this queued message owns retries, not a new
                # webhook publication. Exhausted retries use the shared DLQ.
                raise RuntimeError("Trial notice is temporarily suppressed")

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
                f"Skipping {event.type} for user {event.user_id}: not wanted or "
                "unverified"
            )
            return True

        # The volume knob's own ceiling, across every product notification.
        # Service messages are exempt for the same reason they ignore the rest
        # of the preferences: they are about the customer's account, and a
        # payment failure has to reach them whatever their inbox settings say.
        if event.type not in SERVICE_MESSAGES and not await claim_daily_send(
            event.user_id, preference.daily_limit, _utc_today()
        ):
            logger.info(
                f"Skipping {event.type} for user {event.user_id}: at their "
                f"limit of {preference.daily_limit} emails a day"
            )
            # Acked, not retried: tomorrow's send is a new decision, and a
            # redelivery today would only be refused again.
            return True

        await self.email_sender.send_notification(
            notification_type=event.type,
            user_email=preference.email,
            data=event.data,
            unsubscribe_link=generate_unsubscribe_link(event.user_id),
            volume_links={
                c: generate_preference_link(event.user_id, c) for c in FOOTER_CHOICES
            },
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

    async def _process_pass_work(self, message: str) -> bool:
        """One user's share of a scheduled pass.

        Idempotent by construction: `run_pass_work` claims the user plus the
        period before it does anything, so a redelivery is a no-op rather than
        a second email. A transient failure propagates so the retry loop can
        recover; an unparseable message is permanent and goes to the DLQ.
        """
        try:
            event = PassWorkEvent.model_validate_json(message)
        except ValueError as e:
            logger.warning(f"Unparseable pass work (sending to DLQ): {e}")
            return False
        await briefing_runner.run_pass_work(event)
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
        await channel.set_qos(prefetch_count=CONSUMER_CONCURRENCY)

        # Each consumer declares whether its messages commute, here where a
        # handler is added, rather than in a list somewhere else that a new
        # handler is easy to leave out of. See `Ordering`.
        consumers = {
            USER_NOTIFICATIONS_QUEUE: (
                self._process_user_notification,
                Ordering.COMMUTATIVE,
            ),
            OPS_NOTIFICATIONS_QUEUE: (
                self._process_ops_notification,
                Ordering.COMMUTATIVE,
            ),
            # ADD_CHANGELOG and REMOVE_CHANGELOG for one address are a
            # resubscribe and a churn, and MailerLite is left in whichever
            # state finished last. A removal is a lookup then a delete while
            # an add is one call, so a churn-then-resubscribe pair, one Stripe
            # burst apart, reorders under concurrency and leaves a paying
            # customer out of the changelog. Volume here is one message per
            # subscription lifecycle event, so one at a time costs nothing.
            AUDIENCE_QUEUE: (self._process_audience_change, Ordering.AS_PUBLISHED),
            PASS_WORK_QUEUE: (self._process_pass_work, Ordering.COMMUTATIVE),
        }
        tasks = [
            asyncio.create_task(
                self._consume_queue(
                    await channel.get_queue(name), handler, name, ordering
                )
            )
            for name, (handler, ordering) in consumers.items()
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Service shutdown requested")
            raise
        finally:
            # The four consumers share one channel, so they live and die
            # together. When one fails and `continuous_retry` runs this again,
            # the other three must not be left consuming beside their
            # replacements: every message would then be handled twice.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume_queue(
        self,
        queue: aio_pika.abc.AbstractQueue,
        process_func: Callable[[str], Awaitable[bool]],
        queue_name: str,
        ordering: Ordering,
    ):
        """Work up to this queue's concurrency in messages at once.

        The prefetch already delivered that many; handling them one after
        another only kept the rest waiting in memory. `ordering` is the
        caller's declaration that the queue's messages commute; an
        AS_PUBLISHED queue stays at one message at a time. It has no default
        on purpose, so a new consumer cannot be registered without answering
        the question.
        """
        logger.info(f"Starting consumer for queue: {queue_name}")
        slots = asyncio.Semaphore(ordering.concurrency)
        in_flight: set[asyncio.Task[None]] = set()
        draining = asyncio.Event()

        async def handle(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            try:
                await self._process_message_with_retry(
                    message, process_func, queue_name
                )
            except Exception:
                # Failing here is how the consumer is brought down, but a
                # TaskGroup aborts by cancelling every sibling first and only
                # then the task running the loop, so the grace below would
                # never get a look in: the other messages the prefetch handed
                # this consumer would be cancelled between their send and
                # their ack, and redelivered as second emails. Give them the
                # same bounded grace a shutdown gives, here, before the
                # exception leaves this task and the group aborts. The grace
                # covers only the handlers already running, so the loop must
                # stop starting new ones in the slots they free.
                draining.set()
                siblings = in_flight - {asyncio.current_task()}
                if siblings:
                    await asyncio.wait(siblings, timeout=HANDLER_SHUTDOWN_GRACE_SECONDS)
                raise
            finally:
                slots.release()

        # The TaskGroup is the supervisor: a handler that fails to settle its
        # message (an ack or reject failing for a reason other than a lost
        # channel, which `_settle` absorbs) cancels the loop and surfaces
        # here, so the consumer reconnects instead of sitting alive with
        # unacked deliveries slowly pinning the prefetch. Both ways out --
        # that failure and a shutdown -- reach the handlers as a cancellation,
        # and both give them a bounded grace to settle first.
        try:
            async with asyncio.TaskGroup() as group:
                try:
                    async with queue.iterator() as queue_iter:
                        async for message in queue_iter:
                            if not self.running:
                                break
                            await slots.acquire()
                            if draining.is_set():
                                slots.release()
                                await self._requeue(message, queue_name)
                                # Park until the failing handler's grace is
                                # up and the abort cancels this task. Leaving
                                # the iterator now would cancel the consumer
                                # and nack its buffer mid-grace, and anything
                                # that raised there would abort the grace.
                                await asyncio.Event().wait()
                            task = group.create_task(handle(message))
                            in_flight.add(task)
                            task.add_done_callback(in_flight.discard)
                except asyncio.CancelledError:
                    # Shutdown. Let in-flight handlers settle before the
                    # TaskGroup cancels them, or a send that has already
                    # gone out is redelivered because its ack never went.
                    if in_flight:
                        await asyncio.wait(
                            in_flight, timeout=HANDLER_SHUTDOWN_GRACE_SECONDS
                        )
                    raise
        except asyncio.CancelledError:
            logger.info(f"Consumer for {queue_name} cancelled")
            raise
        except ExceptionGroup as failures:
            # The iterator's exit nacks whatever it still buffered; when the
            # channel has been swapped out under it (a robust reconnect), every
            # one of those fails the same way. The broker requeues unacked
            # deliveries on its own, so this is a reconnect, not a data loss.
            if not _is_channel_loss(failures):
                logger.exception(f"Fatal error in consumer for {queue_name}")
                raise
            logger.warning(f"Consumer for {queue_name} lost its channel; reconnecting")
            raise ConnectionError(f"RabbitMQ channel lost for {queue_name}") from None
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
        write failed) re-runs on retry. The processing timeout is a second,
        quieter way to get one: it cancels the coroutine, but a Postmark send
        runs in a worker thread via ``asyncio.to_thread`` and cancelling the
        await does not stop the thread, so a send that is merely slow can
        still land and then be retried. It takes a stall ten times Postmark's
        own 30s client timeout to reach that, which is why the bound is worth
        having anyway.
        """
        # Only the handler runs inside the retried block. Settling happens in
        # the `else` and after the loop, so an ack or reject that fails for a
        # reason `_settle` does not absorb propagates to the consumer instead
        # of being mistaken for a handler failure and re-running the work.
        last_error: Exception | None = None
        for attempt in range(MAX_CONSUMER_RETRY_ATTEMPTS):
            try:
                body = message.body.decode()
                processed = await asyncio.wait_for(
                    process_func(body), timeout=MESSAGE_PROCESSING_TIMEOUT_SECONDS
                )
            except UnicodeDecodeError as e:
                logger.warning(
                    f"Undecodable message in {queue_name}, sending to DLQ: {e}"
                )
                await self._settle(message, "reject", queue_name)
                return
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"processing exceeded {MESSAGE_PROCESSING_TIMEOUT_SECONDS}s"
                )
            except ClientError as e:
                if not _is_permanent_delivery_failure(e):
                    last_error = e
                else:
                    # A suppressed or malformed address does not become
                    # deliverable by waiting six seconds; retrying only puts
                    # the dead-letter off and logs three warnings for one.
                    logger.warning(
                        f"Permanent delivery failure in {queue_name} "
                        f"(Postmark {e.error_code}), sending to DLQ"
                    )
                    await self._settle(message, "reject", queue_name)
                    return
            except Exception as e:
                last_error = e
            else:
                if processed:
                    await self._settle(message, "ack", queue_name)
                    return
                logger.warning(
                    f"Message in {queue_name} rejected (process_func returned False)"
                )
                await self._settle(message, "reject", queue_name)
                return
            if attempt == MAX_CONSUMER_RETRY_ATTEMPTS - 1:
                break
            delay = CONSUMER_RETRY_BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                f"Transient failure on attempt {attempt + 1}/"
                f"{MAX_CONSUMER_RETRY_ATTEMPTS} in {queue_name}: {last_error}. "
                f"Retrying in {delay}s.",
            )
            await asyncio.sleep(delay)
        logger.exception(
            f"Sending message to DLQ from {queue_name} after "
            f"{MAX_CONSUMER_RETRY_ATTEMPTS} attempts. Last error: {last_error}",
            exc_info=last_error,
        )
        await self._settle(message, "reject", queue_name)

    async def _settle(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
        action: str,
        queue_name: str,
    ) -> None:
        """Ack or dead-letter a message, tolerating a channel that has gone.

        The work is already done by the time this runs. If the channel was
        swapped out underneath (a robust reconnect) the broker has requeued
        the delivery itself; re-running the handler to "retry" the ack would
        be a second send for anything not claimed, and the redelivery is what
        actually settles it. So a lost channel is logged and dropped here,
        never retried.
        """
        try:
            if action == "ack":
                await message.ack()
            else:
                await message.reject(requeue=False)
        except ChannelInvalidStateError:
            logger.warning(
                f"Could not {action} a message in {queue_name}: channel lost, "
                "the broker will redeliver it"
            )

    async def _requeue(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
        queue_name: str,
    ) -> None:
        """Hand back a delivery the consumer pulled but will not work.

        Only reached while the consumer is coming down on another handler's
        failure, so a nack that fails is logged, not raised: raising would
        abort the grace the other handlers are settling in, and an unsettled
        delivery is redelivered once its channel goes anyway.
        """
        try:
            await message.nack(requeue=True)
        except Exception as e:
            logger.warning(
                f"Could not requeue a message in {queue_name}, the broker will "
                f"redeliver it when the channel closes: {e}"
            )

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
