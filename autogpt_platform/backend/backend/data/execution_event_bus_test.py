"""Tests for the sharded channel builders + publish/listen paths on
``AsyncRedisExecutionEventBus`` / ``RedisExecutionEventBus``.

These tests are intentionally Prisma-free: they exercise only the in-process
event-routing layer, using mocks for the Redis cluster client. The live
SSUBSCRIBE round-trip is covered by the integration test in
``event_bus_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import (
    AsyncRedisExecutionEventBus,
    ExecutionEventType,
    ExecutionStatus,
    GraphExecutionEvent,
    NodeExecutionEvent,
    RedisExecutionEventBus,
    _graph_scope_tag,
    exec_channel,
    graph_all_channel,
)

# ---------- Hash-tagged channel builders ----------


def test_graph_scope_tag_uses_hash_tag_syntax():
    """Hash-tagged tag must look like ``{user/graph}`` so per-exec + per-graph
    channels hash to the same Redis Cluster keyslot."""
    assert _graph_scope_tag("u", "g") == "{u/g}"


def test_exec_channel_nests_scope_tag():
    """Per-exec channel: ``{user/graph}/exec/<exec_id>``."""
    assert exec_channel("u", "g", "e") == "{u/g}/exec/e"


def test_graph_all_channel_nests_scope_tag():
    """Aggregate channel: ``{user/graph}/all`` — keyslot-aligned with per-exec."""
    assert graph_all_channel("u", "g") == "{u/g}/all"


def test_exec_and_graph_channels_share_hash_tag():
    """Invariant: both channels *must* share the ``{user/graph}`` prefix.
    If this breaks, SSUBSCRIBE for per-exec and aggregate routes to different
    shards and the per-graph listener loses some events."""
    exec_ch = exec_channel("u", "g", "e")
    graph_ch = graph_all_channel("u", "g")
    assert exec_ch.startswith("{u/g}")
    assert graph_ch.startswith("{u/g}")


# ---------- NodeExecutionEvent publish → exec channel only ----------


def _sample_node_event() -> NodeExecutionEvent:
    now = datetime.now(tz=timezone.utc)
    return NodeExecutionEvent(
        user_id="u",
        graph_id="g",
        graph_version=1,
        graph_exec_id="e",
        node_exec_id="ne",
        node_id="nid",
        block_id="bid",
        status=ExecutionStatus.COMPLETED,
        input_data={"a": 1},
        output_data={"o": [1]},
        add_time=now,
        queue_time=None,
        start_time=now,
        end_time=now,
    )


def _sample_graph_event() -> GraphExecutionEvent:
    now = datetime.now(tz=timezone.utc)
    return GraphExecutionEvent(
        id="e",
        user_id="u",
        graph_id="g",
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


@pytest.mark.asyncio
async def test_async_publish_node_sends_to_both_channels():
    """Node events fan out to BOTH per-exec and aggregate channels so the
    per-graph WS subscriber sees every node update, not just graph-level ones.
    """
    bus = AsyncRedisExecutionEventBus()
    sent_channels: list[str] = []

    async def _capture(self, event, channel_key):
        sent_channels.append(channel_key)

    with patch.object(
        AsyncRedisExecutionEventBus.__mro__[1], "publish_event", _capture
    ):
        await bus._publish_node_exec_update(_sample_node_event())

    assert sent_channels == [
        exec_channel("u", "g", "e"),
        graph_all_channel("u", "g"),
    ]


@pytest.mark.asyncio
async def test_async_publish_graph_sends_to_both_channels():
    bus = AsyncRedisExecutionEventBus()
    sent_channels: list[str] = []

    async def _capture(self, event, channel_key):
        sent_channels.append(channel_key)

    with patch.object(
        AsyncRedisExecutionEventBus.__mro__[1], "publish_event", _capture
    ):
        await bus._publish_graph_exec_update(_sample_graph_event())

    assert sent_channels == [
        exec_channel("u", "g", "e"),
        graph_all_channel("u", "g"),
    ]


@pytest.mark.asyncio
async def test_async_publish_routes_via_type_dispatch():
    """publish() dispatches on the model type — not on status or event_type."""
    bus = AsyncRedisExecutionEventBus()

    with (
        patch.object(bus, "_publish_graph_exec_update", AsyncMock()) as graph_pub,
        patch.object(bus, "_publish_node_exec_update", AsyncMock()) as node_pub,
    ):
        await bus.publish(_sample_graph_event())
        await bus.publish(_sample_node_event())

    graph_pub.assert_awaited_once()
    node_pub.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_publish_truncates_oversized_payload(monkeypatch):
    """Payload truncation applies before sending — size exceeded → replacement."""
    import backend.data.execution as execution

    bus = AsyncRedisExecutionEventBus()
    # Force tiny limit so ``truncate`` rewrites the payload.
    monkeypatch.setattr(execution.config, "max_message_size_limit", 10)

    cluster = MagicMock()
    cluster.execute_command = AsyncMock()
    with patch("backend.data.event_bus.redis.get_redis_async", return_value=cluster):
        await bus.publish(_sample_node_event())

    # Called twice: per-exec and per-graph channel.
    assert cluster.execute_command.await_count == 2


@pytest.mark.asyncio
async def test_async_listen_uses_exec_channel():
    """listen() must subscribe to the per-exec hash-tagged channel."""
    bus = AsyncRedisExecutionEventBus()

    captured: list[str] = []

    async def _listen_events(self, channel_key):
        captured.append(channel_key)
        # Return an empty async-generator so the ``async for`` exits cleanly.
        if False:
            yield  # pragma: no cover

    with patch.object(AsyncRedisExecutionEventBus, "listen_events", _listen_events):
        async for _ in bus.listen("u", "g", "e"):
            break  # pragma: no cover — generator is empty

    assert captured == [exec_channel("u", "g", "e")]


@pytest.mark.asyncio
async def test_async_listen_graph_uses_all_channel():
    """listen_graph() must subscribe to the aggregate hash-tagged channel."""
    bus = AsyncRedisExecutionEventBus()

    captured: list[str] = []

    async def _listen_events(self, channel_key):
        captured.append(channel_key)
        if False:
            yield  # pragma: no cover

    with patch.object(AsyncRedisExecutionEventBus, "listen_events", _listen_events):
        async for _ in bus.listen_graph("u", "g"):
            break  # pragma: no cover — generator is empty

    assert captured == [graph_all_channel("u", "g")]


# ---------- Sync RedisExecutionEventBus (smaller surface; covers branching) ----------


def test_sync_listen_uses_exec_channel():
    bus = RedisExecutionEventBus()

    captured: list[str] = []

    def _listen_events(self, channel_key):
        captured.append(channel_key)
        return iter([])

    with patch.object(RedisExecutionEventBus, "listen_events", _listen_events):
        list(bus.listen("u", "g", "e"))

    assert captured == [exec_channel("u", "g", "e")]


def test_sync_listen_graph_uses_all_channel():
    bus = RedisExecutionEventBus()

    captured: list[str] = []

    def _listen_events(self, channel_key):
        captured.append(channel_key)
        return iter([])

    with patch.object(RedisExecutionEventBus, "listen_events", _listen_events):
        list(bus.listen_graph("u", "g"))

    assert captured == [graph_all_channel("u", "g")]


def test_sync_publish_node_sends_to_both_channels():
    """Sync publish path also fans out to per-exec + per-graph."""
    bus = RedisExecutionEventBus()
    sent: list[str] = []

    def _capture(self, event, channel_key):
        sent.append(channel_key)

    with patch.object(RedisExecutionEventBus.__mro__[1], "publish_event", _capture):
        bus._publish_node_exec_update(_sample_node_event().model_copy())

    assert sent == [
        exec_channel("u", "g", "e"),
        graph_all_channel("u", "g"),
    ]


def test_event_type_is_literal_on_events():
    """event_type is a discriminator literal, not dynamic — the WS fan-out
    relies on ``ExecutionEventType(event_type)`` being stable."""
    node = _sample_node_event()
    graph = _sample_graph_event()
    assert node.event_type == ExecutionEventType.NODE_EXEC_UPDATE
    assert graph.event_type == ExecutionEventType.GRAPH_EXEC_UPDATE


# ---------- Sync publish dispatch + listen yields ----------


def test_sync_publish_dispatches_on_model_type():
    """Sync ``publish()`` routes GraphExecution and NodeExecutionResult to
    their respective helpers — regression guard on the type-dispatch branch."""
    from backend.data.execution import GraphExecution, NodeExecutionResult

    bus = RedisExecutionEventBus()

    graph_like = MagicMock(spec=GraphExecution)
    node_like = MagicMock(spec=NodeExecutionResult)

    with (
        patch.object(bus, "_publish_graph_exec_update") as graph_pub,
        patch.object(bus, "_publish_node_exec_update") as node_pub,
    ):
        bus.publish(graph_like)
        bus.publish(node_like)

    graph_pub.assert_called_once_with(graph_like)
    node_pub.assert_called_once_with(node_like)


def test_sync_publish_graph_exec_update_rebuilds_event():
    """Sync ``_publish_graph_exec_update`` validates the input into a
    GraphExecutionEvent before delegating to ``_publish`` — don't let a raw
    GraphExecution slip through the type-discriminated listener."""
    bus = RedisExecutionEventBus()
    graph_event = _sample_graph_event()

    with patch.object(bus, "_publish") as mock_publish:
        # Feed back the event itself (it's a GraphExecution subclass) to avoid
        # needing a full Graph fixture.
        bus._publish_graph_exec_update(graph_event)

    mock_publish.assert_called_once()
    args = mock_publish.call_args.args
    # The first arg is a GraphExecutionEvent (validated copy).
    assert args[0].event_type == ExecutionEventType.GRAPH_EXEC_UPDATE
    # Channel-routing args match the input.
    assert args[1:] == ("u", "g", "e")


def test_sync_publish_node_exec_update_rebuilds_event():
    """Sync ``_publish_node_exec_update`` validates to NodeExecutionEvent."""
    bus = RedisExecutionEventBus()
    node_event = _sample_node_event()

    with patch.object(bus, "_publish") as mock_publish:
        bus._publish_node_exec_update(node_event)

    mock_publish.assert_called_once()
    args = mock_publish.call_args.args
    assert args[0].event_type == ExecutionEventType.NODE_EXEC_UPDATE
    assert args[1:] == ("u", "g", "e")


def test_sync_publish_graph_truncates_inputs_and_outputs(monkeypatch):
    """Sync ``_publish`` must truncate GraphExecutionEvent.inputs/outputs when
    the payload exceeds the cap — protects Redis from oversized frames."""
    import backend.data.execution as execution

    bus = RedisExecutionEventBus()
    monkeypatch.setattr(execution.config, "max_message_size_limit", 4)

    event = _sample_graph_event()
    event.inputs = {"long": "x" * 10_000}
    event.outputs = {"long": ["y" * 10_000]}

    with patch("backend.data.event_bus.redis.get_redis", return_value=MagicMock()):
        bus._publish(event, "u", "g", "e")

    # After _publish runs, inputs/outputs have been truncated in-place.
    import json as _json

    assert len(_json.dumps(event.inputs)) < 1000
    assert len(_json.dumps(event.outputs)) < 1000


def test_sync_listen_yields_events_from_generator():
    """Sync ``listen()`` must yield through every event produced by the
    underlying ``listen_events`` generator."""
    bus = RedisExecutionEventBus()
    node_ev = _sample_node_event()

    def _listen_events(self, channel_key):
        yield node_ev

    with patch.object(RedisExecutionEventBus, "listen_events", _listen_events):
        got = list(bus.listen("u", "g", "e"))

    assert got == [node_ev]


def test_sync_listen_graph_yields_events_from_generator():
    bus = RedisExecutionEventBus()
    graph_ev = _sample_graph_event()

    def _listen_events(self, channel_key):
        yield graph_ev

    with patch.object(RedisExecutionEventBus, "listen_events", _listen_events):
        got = list(bus.listen_graph("u", "g"))

    assert got == [graph_ev]


def test_execution_bus_name_matches_settings():
    """Both sync and async buses must read the same configured bus name — the
    WS subscriber depends on this for channel naming."""
    assert (
        RedisExecutionEventBus().event_bus_name
        == AsyncRedisExecutionEventBus().event_bus_name
    )
