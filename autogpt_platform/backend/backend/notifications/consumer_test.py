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
    either ends, waits for more like a live queue, or fails on exit."""

    def __init__(
        self,
        messages: list[Any],
        exit_error: BaseException | None = None,
        then_wait: bool = False,
    ):
        self._messages = list(messages)
        self._exit_error = exit_error
        self._then_wait = then_wait

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self._exit_error is not None:
            raise self._exit_error
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._messages:
            return self._messages.pop(0)
        if self._then_wait:
            await asyncio.Event().wait()
        raise StopAsyncIteration


def _queue(
    messages: list[Any],
    exit_error: BaseException | None = None,
    then_wait: bool = False,
) -> MagicMock:
    return MagicMock(iterator=lambda: _FakeIterator(messages, exit_error, then_wait))


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
    # A real queue name, not a placeholder: the audience queue is serialised
    # for ordering and the other three must not be dragged down with it.
    await manager._consume_queue(
        _queue(messages), slow, delivery.USER_NOTIFICATIONS_QUEUE
    )

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


# ── the handlers are supervised ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_settle_failure_is_not_mistaken_for_a_handler_failure():
    """The work succeeded; only the ack failed. Retrying the handler would
    repeat the work, and the retry loop's own DLQ reject would then hide the
    broken channel behind a "sent to DLQ" line."""
    manager = _manager()
    handler = AsyncMock(return_value=True)
    message = _message()
    message.ack.side_effect = RuntimeError("channel is in a bad way")

    with patch.object(delivery.asyncio, "sleep", AsyncMock()) as sleep:
        with pytest.raises(RuntimeError, match="bad way"):
            await manager._process_message_with_retry(message, handler, "q")

    assert handler.await_count == 1
    sleep.assert_not_awaited()
    message.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_handler_that_cannot_settle_takes_the_consumer_down():
    """An ack that fails for a reason `_settle` does not absorb leaves the
    delivery unacked at the broker. Left running, enough of those pin the
    prefetch and the consumer looks alive while delivering nothing; failing
    it is what triggers the reconnect."""
    manager = _manager()
    broken = _message("0")
    broken.ack.side_effect = RuntimeError("channel is in a bad way")
    never = _message("1")

    with pytest.raises(ExceptionGroup) as raised:
        await manager._consume_queue(
            _queue([broken, never], then_wait=True), AsyncMock(return_value=True), "q"
        )

    assert raised.value.subgroup(RuntimeError) is not None


@pytest.mark.asyncio
async def test_shutdown_lets_in_flight_handlers_settle_before_cancelling_them():
    """A handler cancelled between its send and its ack leaves an email
    delivered and the message unacked, which the broker then redelivers."""
    manager = _manager()
    message = _message()

    async def slow(_: str) -> bool:
        await asyncio.sleep(0.05)
        return True

    consumer = asyncio.create_task(
        manager._consume_queue(_queue([message], then_wait=True), slow, "q")
    )
    await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    message.ack.assert_awaited_once()


# ── permanent delivery failures ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("code", "attempts"),
    [(406, 1), (300, 1), (10, delivery.MAX_CONSUMER_RETRY_ATTEMPTS)],
    ids=["inactive-recipient", "invalid-request", "bad-token-is-transient"],
)
@pytest.mark.asyncio
async def test_postmark_rejections_that_cannot_succeed_go_straight_to_the_dlq(
    code: int, attempts: int
):
    """Classified at the retry boundary, so every consumer that sends mail
    gets it, not only the user-notification one."""
    manager = _manager()
    handler = AsyncMock(side_effect=ClientError("no", error_code=code))
    message = _message()

    with patch.object(delivery.asyncio, "sleep", AsyncMock()) as sleep:
        await manager._process_message_with_retry(message, handler, "q")

    assert handler.await_count == attempts
    assert sleep.await_count == attempts - 1
    message.reject.assert_awaited_once_with(requeue=False)
    message.ack.assert_not_awaited()


# ── ordering within a queue ────────────────────────────────────────────────


def _audience(action: delivery.AudienceAction, email: str) -> MagicMock:
    return _message(
        delivery.AudienceEventModel(
            action=action, email=email, user_id="u"
        ).model_dump_json()
    )


@pytest.mark.asyncio
async def test_audience_changes_for_one_email_keep_their_published_order():
    """Concurrency drops FIFO within a queue, and the audience handlers are
    the ones that are not commutative: add_to_changelog and
    remove_from_changelog for the same address leave the subscriber in
    whichever group won the race. Churn then resubscribe is one Stripe burst
    apart, so the two are published back to back."""
    manager = _manager()
    applied: list[str] = []

    async def remove(_: str) -> None:
        # A removal is a lookup and then a delete; an add is one call.
        await asyncio.sleep(0.02)
        applied.append("remove")

    async def add(_: str) -> None:
        applied.append("add")

    messages = [
        _audience(delivery.AudienceAction.REMOVE_CHANGELOG, "churned@example.com"),
        _audience(delivery.AudienceAction.ADD_CHANGELOG, "churned@example.com"),
    ]

    with (
        patch.object(delivery.mailerlite, "remove_from_changelog", remove),
        patch.object(delivery.mailerlite, "add_to_changelog", add),
    ):
        await manager._consume_queue(
            _queue(messages), manager._process_audience_change, delivery.AUDIENCE_QUEUE
        )

    assert applied == ["remove", "add"], (
        "the resubscribe must land after the churn removal it was published "
        "after; reordered, the returning customer is left out of the changelog"
    )
