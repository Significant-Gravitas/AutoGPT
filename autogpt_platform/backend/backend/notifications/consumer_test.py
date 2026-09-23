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
    return MagicMock(
        body=body.encode(), ack=AsyncMock(), reject=AsyncMock(), nack=AsyncMock()
    )


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
    await manager._consume_queue(
        _queue(messages), slow, "q", delivery.Ordering.COMMUTATIVE
    )

    assert peak == delivery.CONSUMER_CONCURRENCY, (
        "the consumer must keep exactly as many messages in flight as the "
        "broker prefetches; fewer wastes the prefetch, more overruns it"
    )
    assert all(m.ack.await_count == 1 for m in messages)


@pytest.mark.asyncio
async def test_an_as_published_queue_works_one_message_at_a_time():
    """The declaration is what serialises the queue, not its name: a queue
    whose messages do not commute must never have two in flight, however many
    the broker prefetched."""
    manager = _manager()
    active = peak = 0

    async def slow(_: str) -> bool:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return True

    messages = [_message() for _ in range(delivery.CONSUMER_CONCURRENCY * 2)]
    await manager._consume_queue(
        _queue(messages), slow, "q", delivery.Ordering.AS_PUBLISHED
    )

    assert peak == 1
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
    await manager._consume_queue(
        _queue(messages), slow, "q", delivery.Ordering.COMMUTATIVE
    )

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
            _queue([], exit_error=group),
            AsyncMock(return_value=True),
            "q",
            delivery.Ordering.COMMUTATIVE,
        )


@pytest.mark.asyncio
async def test_other_exception_groups_still_propagate():
    manager = _manager()
    group = ExceptionGroup("mixed", [ChannelInvalidStateError(), RuntimeError("x")])

    with pytest.raises(ExceptionGroup):
        await manager._consume_queue(
            _queue([], exit_error=group),
            AsyncMock(return_value=True),
            "q",
            delivery.Ordering.COMMUTATIVE,
        )


@pytest.mark.asyncio
async def test_one_failed_consumer_takes_its_siblings_with_it():
    """The four consumers share one channel. If one dies and the service is
    restarted around the other three, every message is handled twice."""
    manager = _manager()
    manager.rabbitmq_config = MagicMock()
    started = asyncio.Event()
    survivors: list[asyncio.Task[None]] = []

    async def consume(
        _queue: Any, _handler: Any, name: str, _ordering: delivery.Ordering
    ) -> None:
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
            _queue([broken, never], then_wait=True),
            AsyncMock(return_value=True),
            "q",
            delivery.Ordering.COMMUTATIVE,
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
        manager._consume_queue(
            _queue([message], then_wait=True), slow, "q", delivery.Ordering.COMMUTATIVE
        )
    )
    await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_does_not_wait_past_the_grace_for_a_handler(monkeypatch):
    """The other half of the grace, and the half with teeth: it is bounded.
    A handler sitting in its retry backoff is up to
    MAX_CONSUMER_RETRY_ATTEMPTS * MESSAGE_PROCESSING_TIMEOUT_SECONDS from
    returning, and `_shutdown_service` waits only SHUTDOWN_TIMEOUT_SECONDS
    for the consumers before it disconnects RabbitMQ. Waiting for such a
    handler to finish on its own puts that disconnect underneath it."""
    monkeypatch.setattr(delivery, "HANDLER_SHUTDOWN_GRACE_SECONDS", 0.05)
    manager = _manager()
    message = _message()
    cancelled = asyncio.Event()

    async def never(_: str) -> bool:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return True

    consumer = asyncio.create_task(
        manager._consume_queue(
            _queue([message], then_wait=True), never, "q", delivery.Ordering.COMMUTATIVE
        )
    )
    await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, timeout=1)

    assert cancelled.is_set()
    # The honest half of the trade: this message is never settled, so the
    # broker redelivers it and anything the handler had already sent goes out
    # twice. The grace narrows that window, it does not close it.
    message.ack.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_settle_failure_lets_its_siblings_finish_before_taking_them_down():
    """Where the two fixes meet, and they do not compose for free. Failing a
    handler is how the consumer is brought down, but a TaskGroup aborts by
    cancelling every sibling first and the task running the loop second, so
    the shutdown grace sits on the wrong side of the abort and never runs.
    Without a grace of its own on this path, one message that cannot be acked
    cancels the nine others the prefetch handed this consumer mid-send, and
    the broker redelivers every one of them as a second email."""
    manager = _manager()
    broken = _message("0")
    broken.ack.side_effect = RuntimeError("channel is in a bad way")
    sibling = _message("1")

    async def work(body: str) -> bool:
        if body == "1":
            await asyncio.sleep(0.05)
        return True

    with pytest.raises(ExceptionGroup) as raised:
        await manager._consume_queue(
            _queue([broken, sibling], then_wait=True),
            work,
            "q",
            delivery.Ordering.COMMUTATIVE,
        )

    assert raised.value.subgroup(RuntimeError) is not None
    sibling.ack.assert_awaited_once()


@pytest.mark.parametrize("requeue_fails", [False, True])
@pytest.mark.asyncio
async def test_no_handler_starts_while_a_settle_failure_waits_on_its_siblings(
    requeue_fails: bool,
):
    """The grace covers the handlers running when the settle failed, and no
    others. Each of those that finishes frees a slot, and a loop still pulling
    would start the next delivery in it, outside the grace: the abort then
    cancels that handler between its send and its ack, and the broker
    redelivers it as a second email. The prefetch is full and more is queued
    behind it, which is how a large fan-out looks when one ack fails. The
    channel that failed the ack may fail the requeue too, and that must not
    cut the grace short either."""
    manager = _manager()
    broken = _message("broken")
    broken.ack.side_effect = RuntimeError("channel is in a bad way")
    fast = _message("fast")
    slow = [_message(f"slow-{i}") for i in range(delivery.CONSUMER_CONCURRENCY - 2)]
    queued = [_message(f"queued-{i}") for i in range(delivery.CONSUMER_CONCURRENCY)]
    if requeue_fails:
        queued[0].nack.side_effect = RuntimeError("channel is in a bad way")
    sent: list[str] = []

    async def send(body: str) -> bool:
        sent.append(body)
        # `fast` frees a slot at once; the slow siblings hold the grace open
        # long enough that anything started in that slot is still mid-send
        # when the grace ends.
        await asyncio.sleep(
            {"broken": 0, "fast": 0.01}.get(body, 0.1 if "slow" in body else 0.5)
        )
        return True

    with pytest.raises(ExceptionGroup) as raised:
        await manager._consume_queue(
            _queue([broken, fast, *slow, *queued], then_wait=True),
            send,
            "q",
            delivery.Ordering.COMMUTATIVE,
        )

    assert raised.value.subgroup(RuntimeError) is not None
    sent_but_unacked = [
        m.body.decode()
        for m in [fast, *slow, *queued]
        if m.body.decode() in sent and not m.ack.await_count
    ]
    assert sent_but_unacked == []
    for sibling in [fast, *slow]:
        sibling.ack.assert_awaited_once()
    # The one delivery the loop had already pulled goes straight back to the
    # queue, rather than sitting unacked on a channel that may outlive it.
    queued[0].nack.assert_awaited_once_with(requeue=True)


@pytest.mark.asyncio
async def test_cancelling_the_service_still_lets_in_flight_handlers_settle():
    """The grace is only worth having if it survives the real shutdown path.
    `_shutdown_service` cancels the service task, not the consumers, and the
    cancel reaches them through `asyncio.gather`, which cancels its children
    and then waits for them before it raises — so `_run_service`'s own
    `finally` cannot land a second cancel on a consumer mid-grace and cut it
    short. Pinned here because none of that is visible from `_consume_queue`
    alone, where the other shutdown tests cancel the consumer directly."""
    manager = _manager()
    manager.rabbitmq_config = MagicMock()
    messages: list[MagicMock] = []

    async def slow(_: str) -> bool:
        await asyncio.sleep(0.05)
        return True

    def next_queue(_name: str) -> MagicMock:
        message = _message()
        messages.append(message)
        return _queue([message], then_wait=True)

    channel = MagicMock(
        set_qos=AsyncMock(), get_queue=AsyncMock(side_effect=next_queue)
    )
    rabbit = MagicMock(connect=AsyncMock(), get_channel=AsyncMock(return_value=channel))

    with (
        patch.object(delivery.rabbitmq, "AsyncRabbitMQ", return_value=rabbit),
        patch.object(manager, "_process_user_notification", slow),
        patch.object(manager, "_process_ops_notification", slow),
        patch.object(manager, "_process_audience_change", slow),
        patch.object(manager, "_process_pass_work", slow),
    ):
        undecorated = inspect.unwrap(NotificationManager._run_service)
        run = asyncio.create_task(undecorated(manager))
        await asyncio.sleep(0.01)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=1)

    assert len(messages) == 4
    for message in messages:
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
            _queue(messages),
            manager._process_audience_change,
            delivery.AUDIENCE_QUEUE,
            delivery.Ordering.AS_PUBLISHED,
        )

    assert applied == ["remove", "add"], (
        "the resubscribe must land after the churn removal it was published "
        "after; reordered, the returning customer is left out of the changelog"
    )


@pytest.mark.asyncio
async def test_every_consumer_declares_how_its_queue_is_ordered():
    """The ordering decision lives next to the handler in `_run_service`, so
    a new consumer cannot be added without making it. Pinned here because
    getting it wrong is silent: the suite stays green and the symptom is a
    rare wrong end state."""
    manager = _manager()
    manager.rabbitmq_config = MagicMock()
    declared: dict[str, delivery.Ordering] = {}

    async def consume(
        _queue: Any, _handler: Any, name: str, ordering: delivery.Ordering
    ) -> None:
        declared[name] = ordering
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
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    assert declared == {
        delivery.USER_NOTIFICATIONS_QUEUE: delivery.Ordering.COMMUTATIVE,
        delivery.OPS_NOTIFICATIONS_QUEUE: delivery.Ordering.COMMUTATIVE,
        delivery.AUDIENCE_QUEUE: delivery.Ordering.AS_PUBLISHED,
        delivery.PASS_WORK_QUEUE: delivery.Ordering.COMMUTATIVE,
    }
