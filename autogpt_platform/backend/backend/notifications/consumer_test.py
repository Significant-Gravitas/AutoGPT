"""The queue consumer under load and under a lost channel.

Messages are worked concurrently up to the prefetch, a lost channel is a
reconnect rather than a retry of work already done, one hung handler cannot
hold the channel past the broker's ack timeout, and a consumer that fails
takes its siblings down with it so a restart cannot double-consume.
"""

import asyncio
import inspect
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aio_pika.exceptions import ChannelInvalidStateError
from postmarker.exceptions import ClientError

from backend.notifications import notifications as delivery
from backend.notifications.notifications import NotificationManager


@pytest.fixture(scope="session")
def server() -> None:
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


def _manager() -> NotificationManager:
    manager = NotificationManager.__new__(NotificationManager)
    manager.running = True
    return manager


def _message(body: str = "{}") -> MagicMock:
    return MagicMock(body=body.encode(), ack=AsyncMock(), reject=AsyncMock())


class _FakeIterator:
    """Stands in for `queue.iterator()`: yields the given messages, then
    behaves on exit however the test says."""

    def __init__(self, messages: list[Any], exit_error: BaseException | None = None):
        self._messages = list(messages)
        self._exit_error = exit_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self._exit_error is not None:
            raise self._exit_error
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _queue(messages: list[Any], exit_error: BaseException | None = None) -> MagicMock:
    return MagicMock(iterator=lambda: _FakeIterator(messages, exit_error))


# ── throughput ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_messages_are_worked_concurrently_up_to_the_prefetch():
    manager = _manager()
    total = delivery.CONSUMER_CONCURRENCY * 3
    active = peak = 0

    async def slow(_: str) -> bool:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return True

    messages = [_message() for _ in range(total)]
    await manager._consume_queue(_queue(messages), slow, "q")

    assert peak == delivery.CONSUMER_CONCURRENCY, (
        "the consumer must keep exactly as many messages in flight as the "
        "broker prefetches; fewer wastes the prefetch, more overruns it"
    )
    assert all(m.ack.await_count == 1 for m in messages)


@pytest.mark.asyncio
async def test_in_flight_work_finishes_before_the_consumer_returns():
    manager = _manager()
    done: list[int] = []

    async def slow(body: str) -> bool:
        await asyncio.sleep(0.01)
        done.append(int(body))
        return True

    messages = [_message(str(i)) for i in range(5)]
    await manager._consume_queue(_queue(messages), slow, "q")

    assert sorted(done) == [0, 1, 2, 3, 4]


# ── a lost channel ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_lost_channel_on_ack_does_not_rerun_the_handler():
    """The work is done; the broker requeues the unacked delivery itself.
    Re-running the handler to 'retry' the ack was three more chances to send
    the same email."""
    manager = _manager()
    handler = AsyncMock(return_value=True)
    message = _message()
    message.ack.side_effect = ChannelInvalidStateError()

    with patch.object(delivery.asyncio, "sleep", AsyncMock()) as sleep:
        await manager._process_message_with_retry(message, handler, "q")

    assert handler.await_count == 1
    message.reject.assert_not_awaited()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_lost_channel_on_reject_is_not_fatal():
    manager = _manager()
    message = _message()
    message.reject.side_effect = ChannelInvalidStateError()

    with patch.object(delivery.asyncio, "sleep", AsyncMock()):
        await manager._process_message_with_retry(
            message, AsyncMock(return_value=False), "q"
        )


@pytest.mark.asyncio
async def test_a_channel_swap_at_iterator_exit_asks_for_a_reconnect():
    """aio_pika's iterator nacks its buffer on exit; on a dead channel that
    fails once per buffered message and surfaces as one ExceptionGroup. That
    is a reconnect, and must not be logged as nine fatal errors."""
    manager = _manager()
    group = ExceptionGroup(
        "Unable to nack all messages", [ChannelInvalidStateError() for _ in range(9)]
    )

    with pytest.raises(ConnectionError, match="channel lost"):
        await manager._consume_queue(
            _queue([], exit_error=group), AsyncMock(return_value=True), "q"
        )


@pytest.mark.asyncio
async def test_other_exception_groups_still_propagate():
    manager = _manager()
    group = ExceptionGroup("mixed", [ChannelInvalidStateError(), RuntimeError("x")])

    with pytest.raises(ExceptionGroup):
        await manager._consume_queue(
            _queue([], exit_error=group), AsyncMock(return_value=True), "q"
        )


@pytest.mark.asyncio
async def test_one_failed_consumer_takes_its_siblings_with_it():
    """The four consumers share one channel. If one dies and the service is
    restarted around the other three, every message is handled twice."""
    manager = _manager()
    manager.rabbitmq_config = MagicMock()
    started = asyncio.Event()
    survivors: list[asyncio.Task[None]] = []

    async def consume(_queue: Any, _handler: Any, name: str) -> None:
        task = asyncio.current_task()
        assert task is not None
        if name == delivery.PASS_WORK_QUEUE:
            await started.wait()
            raise ConnectionError("channel lost")
        survivors.append(task)
        await asyncio.Event().wait()

    channel = MagicMock(
        set_qos=AsyncMock(), get_queue=AsyncMock(return_value=MagicMock())
    )
    rabbit = MagicMock(connect=AsyncMock(), get_channel=AsyncMock(return_value=channel))

    with (
        patch.object(delivery.rabbitmq, "AsyncRabbitMQ", return_value=rabbit),
        patch.object(manager, "_consume_queue", consume),
    ):
        undecorated = inspect.unwrap(NotificationManager._run_service)
        run = asyncio.create_task(undecorated(manager))
        await asyncio.sleep(0.01)
        assert len(survivors) == 3
        started.set()
        with pytest.raises(ConnectionError):
            await run

    assert all(task.cancelled() for task in survivors)


# ── one message cannot hold the channel open ───────────────────────────────


@pytest.mark.asyncio
async def test_a_hung_handler_is_a_transient_failure_not_a_channel_loss(monkeypatch):
    """RabbitMQ closes the channel when a delivery goes unacked for its
    consumer timeout, failing every other in-flight ack with it. A bounded
    wait turns that into one retried message."""
    monkeypatch.setattr(delivery, "MESSAGE_PROCESSING_TIMEOUT_SECONDS", 0.01)
    manager = _manager()
    message = _message()

    async def hang(_: str) -> bool:
        await asyncio.Event().wait()
        return True

    with patch.object(delivery.asyncio, "sleep", AsyncMock()) as sleep:
        await manager._process_message_with_retry(message, hang, "q")

    assert sleep.await_count == delivery.MAX_CONSUMER_RETRY_ATTEMPTS - 1
    message.reject.assert_awaited_once_with(requeue=False)
    message.ack.assert_not_awaited()


# ── permanent delivery failures ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("code", "expect_dlq"),
    [(406, True), (300, True), (10, False)],
    ids=["inactive-recipient", "invalid-request", "bad-token-is-transient"],
)
@pytest.mark.asyncio
async def test_postmark_rejections_that_cannot_succeed_go_straight_to_the_dlq(
    code: int, expect_dlq: bool
):
    manager = _manager()
    manager.email_sender = MagicMock(
        send_notification=AsyncMock(side_effect=ClientError("no", error_code=code))
    )
    preference = MagicMock(email="user@example.com", daily_limit=10)
    db = MagicMock(
        get_user_notification_preference=AsyncMock(return_value=preference),
        get_user_email_verification=AsyncMock(return_value=True),
    )
    parsed = MagicMock(user_id="user-1", type="ALERT", data=MagicMock())

    with (
        patch.object(manager, "_parse_message", return_value=parsed),
        patch.object(delivery, "get_database_manager_async_client", return_value=db),
        patch.object(delivery, "wants_notification", return_value=True),
        patch.object(delivery, "claim_daily_send", AsyncMock(return_value=True)),
        patch.object(delivery, "generate_unsubscribe_link", return_value="u"),
        patch.object(delivery, "generate_preference_link", return_value="p"),
        patch.object(delivery, "SERVICE_MESSAGES", frozenset()),
    ):
        if expect_dlq:
            assert await manager._process_user_notification("{}") is False
        else:
            with pytest.raises(ClientError):
                await manager._process_user_notification("{}")
