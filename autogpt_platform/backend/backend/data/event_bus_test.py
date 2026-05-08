"""Tests for event_bus publish/listen paths."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from backend.data.event_bus import (
    AsyncRedisEventBus,
    RedisEventBus,
    _assert_no_wildcard,
)


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

    try:
        c = redis_client.connect()
    except Exception:  # noqa: BLE001 - any connect failure → skip the test
        return False
    try:
        c.close()
    except Exception:
        pass
    return True


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


def test_sync_listen_events_rejects_wildcard():
    bus = _SyncBusUnderTest()
    with pytest.raises(ValueError):
        next(iter(bus.listen_events("user/*/exec")))


def test_sync_listen_events_ssubscribes_and_yields_decoded_events():
    """Sync listen_events: SSUBSCRIBE on the full channel, decode smessage payloads."""
    bus = _SyncBusUnderTest()

    fake_pubsub = MagicMock()
    fake_pubsub.ssubscribe = MagicMock()
    fake_pubsub.sunsubscribe = MagicMock()
    fake_pubsub.close = MagicMock()
    fake_pubsub.listen = MagicMock(
        return_value=iter(
            [
                {"type": "ssubscribe", "data": 1},
                {"type": "smessage", "data": '{"payload":{"message":"one"}}'},
            ]
        )
    )

    cluster = MagicMock()
    cluster.pubsub = MagicMock(return_value=fake_pubsub)

    with patch("backend.data.event_bus.redis.get_redis", return_value=cluster):
        gen = bus.listen_events("chan")
        first = next(iter(gen))

    assert first.message == "one"
    fake_pubsub.ssubscribe.assert_called_once_with("test_event_bus/chan")


def test_sync_listen_events_teardown_swallows_sunsubscribe_errors():
    """Teardown must not propagate SUNSUBSCRIBE/close failures."""
    bus = _SyncBusUnderTest()

    fake_pubsub = MagicMock()
    fake_pubsub.ssubscribe = MagicMock()
    fake_pubsub.sunsubscribe = MagicMock(side_effect=RuntimeError("SUNSUB broke"))
    fake_pubsub.close = MagicMock(side_effect=RuntimeError("close broke"))
    fake_pubsub.listen = MagicMock(return_value=iter([]))
    cluster = MagicMock()
    cluster.pubsub = MagicMock(return_value=fake_pubsub)

    with patch("backend.data.event_bus.redis.get_redis", return_value=cluster):
        # Exhausting the generator runs the ``finally`` teardown.
        list(bus.listen_events("chan"))

    fake_pubsub.sunsubscribe.assert_called_once()
    fake_pubsub.close.assert_called_once()


# ---------- Async close() teardown ----------


@pytest.mark.asyncio
async def test_async_close_is_noop():
    """close() is a backward-compat no-op now that listen_events owns its own state."""
    bus = _BusUnderTest()
    # Repeated calls must not crash; pubsub/client are generator-locals.
    await bus.close()
    await bus.close()


@pytest.mark.asyncio
async def test_async_listen_events_swallows_aclose_errors():
    """Broken pubsub.aclose / client.aclose must not propagate to the caller."""
    bus = _BusUnderTest()

    fake_pubsub = MagicMock()
    fake_pubsub.execute_command = AsyncMock()
    fake_pubsub.channels = {}
    fake_pubsub.aclose = AsyncMock(side_effect=RuntimeError("pubsub broke"))

    async def _listen():
        return
        yield  # pragma: no cover — unreachable

    fake_pubsub.listen = _listen

    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock(side_effect=RuntimeError("client broke"))

    with patch(
        "backend.data.event_bus.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        async for _ in bus.listen_events("chan"):
            pass  # pragma: no cover — never yields

    # Both aclose attempts must have run despite raising.
    fake_pubsub.aclose.assert_awaited_once()
    fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_listen_events_concurrent_does_not_share_state():
    """Two concurrent listens on the same bus must keep their pubsub/client local."""
    bus = _BusUnderTest()

    pubsubs: list[MagicMock] = []
    clients: list[MagicMock] = []
    started = asyncio.Event()
    proceed = asyncio.Event()

    def _make_pair() -> tuple[MagicMock, MagicMock]:
        pubsub = MagicMock()
        pubsub.execute_command = AsyncMock()
        pubsub.channels = {}
        pubsub.aclose = AsyncMock()

        async def _listen():
            started.set()
            await proceed.wait()
            return
            yield  # pragma: no cover — unreachable

        pubsub.listen = _listen

        client = MagicMock()
        client.pubsub = MagicMock(return_value=pubsub)
        client.aclose = AsyncMock()
        pubsubs.append(pubsub)
        clients.append(client)
        return pubsub, client

    async def _factory(_chan: str):
        _, client = _make_pair()
        return client

    with patch(
        "backend.data.event_bus.redis.connect_sharded_pubsub_async",
        AsyncMock(side_effect=_factory),
    ):

        async def _run():
            async for _ in bus.listen_events("chan"):
                pass  # pragma: no cover — never yields

        task_a = asyncio.create_task(_run())
        task_b = asyncio.create_task(_run())
        # Wait for both pumps to be parked inside listen() before unblocking.
        await started.wait()
        # Yield once more so the second task also enters listen().
        await asyncio.sleep(0)
        proceed.set()
        await asyncio.gather(task_a, task_b)

    # Each listen must have closed its OWN pubsub/client exactly once. If
    # either was closed twice or zero times, the singleton race is back.
    assert len(pubsubs) == 2
    for pubsub in pubsubs:
        pubsub.aclose.assert_awaited_once()
    for client in clients:
        client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_wait_for_event_returns_none_on_timeout():
    """wait_for_event must coerce asyncio.TimeoutError → None."""
    import asyncio as _asyncio

    bus = _BusUnderTest()

    async def _never(self, channel_key):
        await _asyncio.sleep(10)
        yield  # pragma: no cover — unreachable

    with patch.object(_BusUnderTest, "listen_events", _never):
        result = await bus.wait_for_event("chan", timeout=0.01)

    assert result is None


# The listen_events async happy path is covered by the live-cluster integration
# test above; this one exercises the close-on-exception fallback.
@pytest.mark.asyncio
async def test_async_listen_events_closes_on_exception():
    """If the pump raises, close() must still run to release the shard-pinned client."""
    bus = _BusUnderTest()

    fake_pubsub = MagicMock()
    fake_pubsub.execute_command = AsyncMock()
    fake_pubsub.channels = {}
    fake_pubsub.aclose = AsyncMock()

    class _Boom(Exception):
        pass

    async def _listen():
        raise _Boom()
        yield  # pragma: no cover — unreachable

    fake_pubsub.listen = _listen

    fake_client = MagicMock()
    fake_client.pubsub = MagicMock(return_value=fake_pubsub)
    fake_client.aclose = AsyncMock()

    with patch(
        "backend.data.event_bus.redis.connect_sharded_pubsub_async",
        AsyncMock(return_value=fake_client),
    ):
        with pytest.raises(_Boom):
            async for _ in bus.listen_events("chan"):
                pass

    # close() must have fired (both aclose calls).
    fake_pubsub.aclose.assert_awaited_once()
    fake_client.aclose.assert_awaited_once()
