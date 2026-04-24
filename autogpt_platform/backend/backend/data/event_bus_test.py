"""Tests for event_bus publish/listen paths.

Focus on SPUBLISH routing, dual-publish during migration, and the
wildcard guard that replaces the old PSUBSCRIBE-based wildcard.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from backend.data.event_bus import AsyncRedisEventBus, _assert_no_wildcard


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
    """publish_event routes SPUBLISH + classic PUBLISH through the cluster client."""
    bus = _BusUnderTest()
    event = SampleEvent(message="test message")

    mock_cluster = MagicMock()
    mock_cluster.execute_command = AsyncMock()

    with (
        patch(
            "backend.data.event_bus.redis.get_redis_async", return_value=mock_cluster
        ),
        patch("backend.data.event_bus.DUAL_PUBLISH", True),
    ):
        await bus.publish_event(event, "test_channel")

    # Both SPUBLISH and classic PUBLISH fire via the cluster client while
    # dual-publish is on.
    assert mock_cluster.execute_command.await_count == 2
    commands = [call.args[0] for call in mock_cluster.execute_command.await_args_list]
    assert commands == ["SPUBLISH", "PUBLISH"]


@pytest.mark.asyncio
async def test_publish_event_skips_classic_when_dual_publish_off():
    """When dual-publish is off, only SPUBLISH runs."""
    bus = _BusUnderTest()
    event = SampleEvent(message="test message")

    mock_cluster = MagicMock()
    mock_cluster.execute_command = AsyncMock()

    with (
        patch(
            "backend.data.event_bus.redis.get_redis_async", return_value=mock_cluster
        ),
        patch("backend.data.event_bus.DUAL_PUBLISH", False),
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


# Live SSUBSCRIBE round-trip against the 3-shard compose cluster. Skip
# cleanly when no cluster is reachable so CI without docker doesn't flap.


def _has_live_cluster() -> bool:
    from backend.data import redis_client

    try:
        c = redis_client.connect()
    except Exception:  # noqa: BLE001
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
    # Let SSUBSCRIBE settle before publishing; ssubscribe is sent before
    # Redis can ack, and publishes racing the subscribe are dropped.
    await asyncio.sleep(0.3)
    try:
        with patch("backend.data.event_bus.DUAL_PUBLISH", False):
            await publisher.publish_event(
                SampleEvent(message="hello-ssub"), channel_key
            )
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
        with patch("backend.data.event_bus.DUAL_PUBLISH", False):
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
