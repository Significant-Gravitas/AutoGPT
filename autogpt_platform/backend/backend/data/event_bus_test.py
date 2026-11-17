"""Tests for event_bus publish/listen paths."""

import asyncio
from contextlib import aclosing
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

import backend.data.event_bus as event_bus
from backend.data.event_bus import (
    AsyncRedisEventBus,
    RedisEventBus,
    _assert_no_wildcard,
)
from backend.data.sharded_pubsub import EventStreamLostError, ShardedSubscription
from backend.util.testing import is_tcp_port_reachable


class SampleEvent(BaseModel):
    """Minimal event model used by the tests below."""

    message: str


class _BusUnderTest(AsyncRedisEventBus[SampleEvent]):
    Model = SampleEvent

    @property
    def event_bus_name(self) -> str:
        return "test_event_bus"


@pytest.mark.asyncio
async def test_publish_event_handles_connection_failure_gracefully():
    """publish_event must log and swallow when the cluster client is down."""
    bus = _BusUnderTest()
    event = SampleEvent(message="test message")

    with patch(
        "backend.data.event_bus.redis.get_redis_async",
        side_effect=ConnectionError("Authentication required."),
    ):
        # Should not raise exception
        await bus.publish_event(event, "test_channel")


@pytest.mark.asyncio
async def test_publish_event_spublishes_via_cluster_client():
    """publish_event routes a single SPUBLISH through the cluster client."""
    bus = _BusUnderTest()
    event = SampleEvent(message="test message")

    mock_cluster = MagicMock()
    mock_cluster.execute_command = AsyncMock()

    with patch(
        "backend.data.event_bus.redis.get_redis_async", return_value=mock_cluster
    ):
        await bus.publish_event(event, "test_channel")

    mock_cluster.execute_command.assert_awaited_once()
    assert mock_cluster.execute_command.await_args[0][0] == "SPUBLISH"


@pytest.mark.asyncio
async def test_publish_event_rejects_wildcard_channel():
    """A channel_key containing ``*`` must raise — no silent no-op."""
    bus = _BusUnderTest()
    with patch("backend.data.event_bus.redis.get_redis_async") as get_cluster:
        with pytest.raises(ValueError):
            await bus.publish_event(SampleEvent(message="m"), "user/*/exec")
    # The cluster client must never be reached for a wildcard channel.
    get_cluster.assert_not_called()


def test_assert_no_wildcard_guard():
    """The standalone guard must reject any ``*``-containing channel."""
    with pytest.raises(ValueError):
        _assert_no_wildcard("user/*/exec")
    # Concrete channels must pass.
    _assert_no_wildcard("execution_event/user-1/graph-1/exec-1")


# Live SSUBSCRIBE round-trip; skipped when no cluster is reachable.


def _has_live_cluster() -> bool:
    from backend.data import redis_client

    return is_tcp_port_reachable(redis_client.HOST, redis_client.PORT)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip SSUBSCRIBE integration",
)
async def test_ssubscribe_end_to_end_async():
    """SPUBLISH on one AsyncRedisEventBus reaches SSUBSCRIBE on another."""
    import asyncio

    from backend.data import redis_client

    redis_client.get_redis.cache_clear()
    redis_client._async_clients.clear()

    publisher = _BusUnderTest()
    subscriber = _BusUnderTest()
    channel_key = "pr12900:event_bus:integration"

    received: list[SampleEvent] = []

    async def consume() -> None:
        async for ev in subscriber.listen_events(channel_key):
            received.append(ev)
            return

    task = asyncio.create_task(consume())
    # Let SSUBSCRIBE settle; races drop the publish otherwise.
    await asyncio.sleep(0.3)
    try:
        await publisher.publish_event(SampleEvent(message="hello-ssub"), channel_key)
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        if not task.done():
            task.cancel()
        await subscriber.close()
        await redis_client.disconnect_async()

    assert received and received[0].message == "hello-ssub"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip execution-bus integration",
)
async def test_execution_bus_listen_and_listen_graph_both_deliver():
    """Per-exec and per-graph channels both receive every execution event."""
    import asyncio
    from datetime import datetime, timezone

    from backend.data import redis_client
    from backend.data.execution import (
        AsyncRedisExecutionEventBus,
        ExecutionStatus,
        GraphExecutionEvent,
    )

    redis_client.get_redis.cache_clear()
    redis_client._async_clients.clear()

    user_id = "user-it"
    graph_id = "graph-it"
    exec_id = "exec-it"
    now = datetime.now(tz=timezone.utc)

    event = GraphExecutionEvent(
        id=exec_id,
        user_id=user_id,
        graph_id=graph_id,
        graph_version=1,
        preset_id=None,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        ended_at=now,
        stats=GraphExecutionEvent.Stats(
            cost=0, duration=0.1, node_exec_time=0.1, node_exec_count=1
        ),
        inputs={},
        credential_inputs=None,
        nodes_input_masks=None,
        outputs={},
    )

    single = AsyncRedisExecutionEventBus()
    all_execs = AsyncRedisExecutionEventBus()
    publisher = AsyncRedisExecutionEventBus()

    received_single: list = []
    received_all: list = []

    async def _listen_single() -> None:
        async for ev in single.listen(user_id, graph_id, exec_id):
            received_single.append(ev)
            return

    async def _listen_all() -> None:
        async for ev in all_execs.listen_graph(user_id, graph_id):
            received_all.append(ev)
            return

    t1 = asyncio.create_task(_listen_single())
    t2 = asyncio.create_task(_listen_all())
    await asyncio.sleep(0.3)

    try:
        await publisher.publish(event)
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5.0)
    finally:
        for t in (t1, t2):
            if not t.done():
                t.cancel()
        await single.close()
        await all_execs.close()
        await publisher.close()
        await redis_client.disconnect_async()

    assert received_single and received_single[0].id == exec_id
    assert received_all and received_all[0].id == exec_id


@pytest.mark.asyncio
async def test_listen_events_rejects_wildcard_channel():
    """listen_events on a wildcard channel must raise before touching Redis."""
    bus = _BusUnderTest()
    with pytest.raises(ValueError):
        async for _ in bus.listen_events("user/*/exec"):
            break


# ---------- Serialization + size guard ----------


def test_serialize_message_tags_full_channel_name():
    """_serialize_message returns the ``<bus>/<key>`` full channel name."""
    bus = _BusUnderTest()
    _, full = bus._serialize_message(SampleEvent(message="x"), "chan")
    assert full == "test_event_bus/chan"


def test_serialize_message_truncates_oversized_payload(monkeypatch):
    """If the payload exceeds max_message_size_limit, it's replaced with an
    ``error_comms_update`` payload rather than crashing the cluster."""
    import backend.data.event_bus as event_bus

    bus = _BusUnderTest()
    # Cap tiny to force truncation.
    monkeypatch.setattr(event_bus.config, "max_message_size_limit", 50)
    message, _ = bus._serialize_message(SampleEvent(message="x" * 1000), "chan")
    assert "error_comms_update" in message
    assert "Payload too large" in message


def test_deserialize_message_rejects_non_pubsub_types():
    """Non ``smessage|message|pmessage`` deliveries deserialize to None."""
    bus = _BusUnderTest()
    assert bus._deserialize_message({"type": "ssubscribe", "data": 1}, "c") is None
    assert bus._deserialize_message({"type": "subscribe", "data": 1}, "c") is None


def test_deserialize_message_swallows_bad_json():
    """Corrupted payloads must not raise — they return None (logged elsewhere)."""
    bus = _BusUnderTest()
    assert (
        bus._deserialize_message({"type": "smessage", "data": "not-json"}, "c") is None
    )


def test_deserialize_message_parses_smessage():
    """Happy-path ``smessage`` yields the inner event model."""
    bus = _BusUnderTest()
    wrapped = '{"payload":{"message":"hi"}}'
    parsed = bus._deserialize_message({"type": "smessage", "data": wrapped}, "chan")
    assert parsed is not None and parsed.message == "hi"


# ---------- Sync RedisEventBus ----------


class _SyncBusUnderTest(RedisEventBus[SampleEvent]):
    Model = SampleEvent

    @property
    def event_bus_name(self) -> str:
        return "test_event_bus"


def test_sync_publish_event_spublish_only():
    """Sync publish_event must issue a single SPUBLISH (no classic fallback)."""
    bus = _SyncBusUnderTest()
    cluster = MagicMock()
    cluster.execute_command = MagicMock()

    with patch("backend.data.event_bus.redis.get_redis", return_value=cluster):
        bus.publish_event(SampleEvent(message="m"), "chan")

    cluster.execute_command.assert_called_once()
    assert cluster.execute_command.call_args.args[0] == "SPUBLISH"


def test_sync_publish_event_rejects_wildcard():
    bus = _SyncBusUnderTest()
    with patch("backend.data.event_bus.redis.get_redis") as mock_get:
        with pytest.raises(ValueError):
            bus.publish_event(SampleEvent(message="m"), "user/*/exec")
    mock_get.assert_not_called()


def test_sync_publish_event_swallows_connection_errors():
    """publish_event must never raise to callers — logs + drops on failure."""
    bus = _SyncBusUnderTest()
    with patch(
        "backend.data.event_bus.redis.get_redis",
        side_effect=ConnectionError("no redis"),
    ):
        # Should NOT raise.
        bus.publish_event(SampleEvent(message="m"), "chan")


# ---------- Async close() teardown ----------


@pytest.mark.asyncio
async def test_async_close_is_noop():
    """close() is a backward-compat no-op now that listen_events owns its own state."""
    bus = _BusUnderTest()
    # Repeated calls must not crash; pubsub/client are generator-locals.
    await bus.close()
    await bus.close()


class _FakeSubscription:
    """Stands in for ``ShardedSubscription``: yields ``messages`` in order,
    then whatever ``end`` says (by default, waits like a quiet stream)."""

    def __init__(self, messages: list[dict], end: Exception | None = None) -> None:
        self._messages = messages
        self._end = end
        self.closed = 0

    async def messages(self):
        for message in self._messages:
            yield message
        if self._end is not None:
            raise self._end
        await asyncio.Event().wait()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.closed += 1


def _smessage(text: str) -> dict:
    return {"type": "smessage", "data": f'{{"payload":{{"message":"{text}"}}}}'}


@pytest.mark.asyncio
async def test_async_listen_events_opens_its_own_subscription_per_listen():
    """Concurrent listens on one bus (the singleton webhook bus) must not share
    a connection: each opens, and closes, its own."""
    bus = _BusUnderTest()
    opened: list[_FakeSubscription] = []

    async def open_(channel: str) -> _FakeSubscription:
        opened.append(_FakeSubscription([_smessage(channel)]))
        return opened[-1]

    with patch.object(ShardedSubscription, "open", side_effect=open_):
        async with (
            aclosing(bus.listen_events("a")) as a,
            aclosing(bus.listen_events("b")) as b,
        ):
            first, second = await asyncio.gather(anext(a), anext(b))

    assert {first.message, second.message} == {"test_event_bus/a", "test_event_bus/b"}
    assert len(opened) == 2


@pytest.mark.asyncio
async def test_async_listen_events_closes_its_subscription_when_lost():
    bus = _BusUnderTest()
    subscription = _FakeSubscription([], end=EventStreamLostError("gone"))

    with patch.object(
        ShardedSubscription, "open", AsyncMock(return_value=subscription)
    ):
        with pytest.raises(EventStreamLostError):
            async for _ in bus.listen_events("chan"):
                pass

    assert subscription.closed == 1


@pytest.mark.asyncio
async def test_async_listen_events_reports_a_shard_dropping_the_subscription():
    """Redis unsubscribes us itself when the channel's slot moves to another
    shard; nothing more will arrive on this connection."""
    bus = _BusUnderTest()
    subscription = _FakeSubscription(
        [_smessage("before"), {"type": "sunsubscribe", "channel": "x", "data": 0}]
    )
    got: list[str] = []

    with patch.object(
        ShardedSubscription, "open", AsyncMock(return_value=subscription)
    ):
        with pytest.raises(EventStreamLostError):
            async for event in bus.listen_events("chan"):
                got.append(event.message)

    assert got == ["before"]


@pytest.mark.asyncio
async def test_listen_with_resync_marks_each_subscription_and_each_gap_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """``None`` after each confirmed subscription and once per gap, so a caller
    re-reads state whenever events may have been missed, but not again for
    every failed attempt while Redis stays down."""
    monkeypatch.setattr(event_bus, "RESUBSCRIBE_DELAY_S", 0)
    bus = _BusUnderTest()
    attempts = [
        _FakeSubscription([_smessage("one")], end=EventStreamLostError("dropped")),
        EventStreamLostError("redis down"),
        EventStreamLostError("redis still down"),
        _FakeSubscription([_smessage("two")]),
    ]

    async def open_(_channel: str) -> _FakeSubscription:
        attempt = attempts.pop(0)
        if isinstance(attempt, Exception):
            raise attempt
        return attempt

    seen: list[str | None] = []
    with patch.object(ShardedSubscription, "open", side_effect=open_):
        async with aclosing(bus.listen_with_resync("chan")) as events:
            async for event in events:
                seen.append(None if event is None else event.message)
                if event is not None and event.message == "two":
                    break

    assert seen == [None, "one", None, None, "two"]


@pytest.mark.asyncio
async def test_a_shard_that_drops_each_new_subscription_is_backed_off(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only a stream that stayed up resets the delay; otherwise a flapping
    shard would make every listener resync about once a second."""
    pauses = AsyncMock()
    monkeypatch.setattr(event_bus, "sleep", pauses)
    monkeypatch.setattr(event_bus, "RESUBSCRIBE_DELAY_S", 1.0)
    bus = _BusUnderTest()

    async def open_(_channel: str) -> _FakeSubscription:
        return _FakeSubscription([], end=EventStreamLostError("dropped at once"))

    with patch.object(ShardedSubscription, "open", side_effect=open_):
        async with aclosing(bus.listen_with_resync("chan")) as events:
            for _ in range(10):  # five cycles: a mark at each open and loss
                await anext(events)

    delays = [call.args[0] for call in pauses.await_args_list]
    # 1, 2, 4, 8 s with +/-50 % jitter; a reset at each open stays at <= 1.5 s.
    assert len(delays) == 4 and delays[3] >= 4.0


@pytest.mark.asyncio
async def test_a_stream_that_stayed_up_starts_the_backoff_over(
    monkeypatch: pytest.MonkeyPatch,
):
    pauses = AsyncMock()
    monkeypatch.setattr(event_bus, "sleep", pauses)
    monkeypatch.setattr(event_bus, "RESUBSCRIBE_DELAY_S", 1.0)
    monkeypatch.setattr(event_bus, "RESUBSCRIBE_HEALTHY_S", 0.0)
    bus = _BusUnderTest()

    async def open_(_channel: str) -> _FakeSubscription:
        return _FakeSubscription([], end=EventStreamLostError("dropped"))

    with patch.object(ShardedSubscription, "open", side_effect=open_):
        async with aclosing(bus.listen_with_resync("chan")) as events:
            for _ in range(10):
                await anext(events)

    delays = [call.args[0] for call in pauses.await_args_list]
    assert len(delays) == 4 and max(delays) <= 1.5


@pytest.mark.asyncio
async def test_async_wait_for_event_returns_none_on_timeout():
    """wait_for_event must coerce asyncio.TimeoutError → None."""
    bus = _BusUnderTest()

    async def _never(self, channel_key):
        await asyncio.sleep(10)
        yield  # pragma: no cover — unreachable

    with patch.object(_BusUnderTest, "listen_events", _never):
        result = await bus.wait_for_event("chan", timeout=0.01)

    assert result is None


@pytest.mark.asyncio
async def test_async_wait_for_event_returns_none_when_the_stream_is_lost():
    bus = _BusUnderTest()
    subscription = _FakeSubscription([], end=EventStreamLostError("gone"))

    with patch.object(
        ShardedSubscription, "open", AsyncMock(return_value=subscription)
    ):
        assert await bus.wait_for_event("chan", timeout=5) is None
