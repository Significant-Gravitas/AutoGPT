"""ConnectionManager integration over the live 3-shard Redis cluster:
SSUBSCRIBE → SPUBLISH → WebSocket forwarding with no Redis mocks. Skips
when the cluster is unreachable."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import WebSocket

import backend.data.redis_client as redis_client
from backend.api.conn_manager import (
    ConnectionManager,
    _graph_execs_channel_key,
    event_bus_channel,
    graph_exec_channel_key,
)
from backend.api.model import WSMethod
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEvent,
    GraphExecutionMeta,
    NodeExecutionEvent,
    exec_channel,
    graph_all_channel,
)


def _has_live_cluster() -> bool:
    try:
        c = redis_client.connect()
    except Exception:  # noqa: BLE001 — any connect failure → skip
        return False
    try:
        c.close()
    except Exception:
        pass
    return True


pytestmark = pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip conn_manager integration",
)


def _meta(user_id: str, graph_id: str, graph_exec_id: str) -> GraphExecutionMeta:
    """Build a minimal GraphExecutionMeta for ``subscribe_graph_exec`` to use."""
    return GraphExecutionMeta(
        id=graph_exec_id,
        user_id=user_id,
        graph_id=graph_id,
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=ExecutionStatus.RUNNING,
        started_at=datetime.now(tz=timezone.utc),
        ended_at=None,
        stats=GraphExecutionMeta.Stats(),
    )


def _node_event_payload(
    *, user_id: str, graph_id: str, graph_exec_id: str, marker: str
) -> bytes:
    """Wire-format a NodeExecutionEvent the way RedisExecutionEventBus would."""
    inner = NodeExecutionEvent(
        user_id=user_id,
        graph_id=graph_id,
        graph_version=1,
        graph_exec_id=graph_exec_id,
        node_exec_id=f"node-exec-{marker}",
        node_id="node-1",
        block_id="block-1",
        status=ExecutionStatus.COMPLETED,
        input_data={"in": marker},
        output_data={"out": [marker]},
        add_time=datetime.now(tz=timezone.utc),
        queue_time=None,
        start_time=datetime.now(tz=timezone.utc),
        end_time=datetime.now(tz=timezone.utc),
    ).model_dump(mode="json")
    return json.dumps({"payload": inner}).encode()


def _graph_event_payload(
    *, user_id: str, graph_id: str, graph_exec_id: str, marker: str
) -> bytes:
    inner = GraphExecutionEvent(
        id=graph_exec_id,
        user_id=user_id,
        graph_id=graph_id,
        graph_version=1,
        preset_id=None,
        status=ExecutionStatus.COMPLETED,
        started_at=datetime.now(tz=timezone.utc),
        ended_at=datetime.now(tz=timezone.utc),
        stats=GraphExecutionEvent.Stats(
            cost=0,
            duration=1.0,
            node_exec_time=0.5,
            node_exec_count=1,
        ),
        inputs={"x": marker},
        credential_inputs=None,
        nodes_input_masks=None,
        outputs={"y": [marker]},
    ).model_dump(mode="json")
    return json.dumps({"payload": inner}).encode()


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll ``predicate()`` until truthy or timeout — used to wait for pubsub."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_two_clients_get_independent_ssubscribes_on_right_shards(
    monkeypatch,
) -> None:
    """Two WS clients on different graph_exec_ids each receive ONLY their
    own publish, even when the channels land on different shards."""
    user_id = "user-conn-int-1"
    graph_a = f"graph-a-{uuid4().hex[:8]}"
    graph_b = f"graph-b-{uuid4().hex[:8]}"
    exec_a = f"exec-a-{uuid4().hex[:8]}"
    exec_b = f"exec-b-{uuid4().hex[:8]}"

    # Stub Prisma lookup so tests don't need a DB.
    async def _fake_meta(_uid, gex_id):
        return _meta(user_id, graph_a if gex_id == exec_a else graph_b, gex_id)

    monkeypatch.setattr("backend.api.conn_manager.get_graph_execution_meta", _fake_meta)

    cm = ConnectionManager()
    ws_a: AsyncMock = AsyncMock(spec=WebSocket)
    ws_b: AsyncMock = AsyncMock(spec=WebSocket)
    sent_a: list[str] = []
    sent_b: list[str] = []
    ws_a.send_text = AsyncMock(side_effect=lambda m: sent_a.append(m))
    ws_b.send_text = AsyncMock(side_effect=lambda m: sent_b.append(m))

    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()

    try:
        await cm.subscribe_graph_exec(
            user_id=user_id, graph_exec_id=exec_a, websocket=ws_a
        )
        await cm.subscribe_graph_exec(
            user_id=user_id, graph_exec_id=exec_b, websocket=ws_b
        )
        # Let SSUBSCRIBE settle on each shard.
        await asyncio.sleep(0.2)

        # Publish to each per-exec channel.
        chan_a = event_bus_channel(exec_channel(user_id, graph_a, exec_a))
        chan_b = event_bus_channel(exec_channel(user_id, graph_b, exec_b))
        cluster.spublish(
            chan_a,
            _node_event_payload(
                user_id=user_id,
                graph_id=graph_a,
                graph_exec_id=exec_a,
                marker="A",
            ).decode(),
        )
        cluster.spublish(
            chan_b,
            _node_event_payload(
                user_id=user_id,
                graph_id=graph_b,
                graph_exec_id=exec_b,
                marker="B",
            ).decode(),
        )

        delivered = await _wait_until(lambda: sent_a and sent_b, timeout=5.0)
        assert delivered, f"timeout: sent_a={sent_a!r} sent_b={sent_b!r}"

        msg_a = json.loads(sent_a[0])
        msg_b = json.loads(sent_b[0])
        assert msg_a["channel"] == graph_exec_channel_key(user_id, graph_exec_id=exec_a)
        assert msg_b["channel"] == graph_exec_channel_key(user_id, graph_exec_id=exec_b)
        assert msg_a["data"]["graph_exec_id"] == exec_a
        assert msg_b["data"]["graph_exec_id"] == exec_b
        # No cross-talk: each socket got exactly one message.
        assert len(sent_a) == 1 and len(sent_b) == 1
    finally:
        await cm.disconnect_socket(ws_a, user_id=user_id)
        await cm.disconnect_socket(ws_b, user_id=user_id)
        redis_client.disconnect()


@pytest.mark.asyncio
async def test_aggregate_channel_receives_per_exec_publishes(monkeypatch) -> None:
    """A subscriber on the ``graph_execs`` aggregate channel must receive the
    GraphExecutionEvent published to the ``/all`` channel — even though
    per-exec events go to a different channel."""
    user_id = "user-conn-int-2"
    graph_id = f"graph-{uuid4().hex[:8]}"
    exec_id = f"exec-{uuid4().hex[:8]}"

    async def _fake_meta(_uid, gex_id):
        return _meta(user_id, graph_id, gex_id)

    monkeypatch.setattr("backend.api.conn_manager.get_graph_execution_meta", _fake_meta)

    cm = ConnectionManager()
    ws_agg: AsyncMock = AsyncMock(spec=WebSocket)
    ws_per: AsyncMock = AsyncMock(spec=WebSocket)
    sent_agg: list[str] = []
    sent_per: list[str] = []
    ws_agg.send_text = AsyncMock(side_effect=lambda m: sent_agg.append(m))
    ws_per.send_text = AsyncMock(side_effect=lambda m: sent_per.append(m))

    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()

    try:
        await cm.subscribe_graph_execs(
            user_id=user_id, graph_id=graph_id, websocket=ws_agg
        )
        await cm.subscribe_graph_exec(
            user_id=user_id, graph_exec_id=exec_id, websocket=ws_per
        )
        await asyncio.sleep(0.2)

        # The eventbus publishes the same event to both channels — replicate.
        chan_per = event_bus_channel(exec_channel(user_id, graph_id, exec_id))
        chan_all = event_bus_channel(graph_all_channel(user_id, graph_id))
        payload = _graph_event_payload(
            user_id=user_id,
            graph_id=graph_id,
            graph_exec_id=exec_id,
            marker="agg",
        ).decode()
        cluster.spublish(chan_per, payload)
        cluster.spublish(chan_all, payload)

        delivered = await _wait_until(lambda: sent_agg and sent_per, timeout=5.0)
        assert delivered, f"sent_agg={sent_agg!r} sent_per={sent_per!r}"
        agg_msg = json.loads(sent_agg[0])
        per_msg = json.loads(sent_per[0])
        # Aggregate subscriber's channel key is the per-graph executions key.
        assert agg_msg["channel"] == _graph_execs_channel_key(
            user_id, graph_id=graph_id
        )
        assert per_msg["channel"] == graph_exec_channel_key(
            user_id, graph_exec_id=exec_id
        )
        assert agg_msg["method"] == WSMethod.GRAPH_EXECUTION_EVENT.value
    finally:
        await cm.disconnect_socket(ws_agg, user_id=user_id)
        await cm.disconnect_socket(ws_per, user_id=user_id)
        redis_client.disconnect()


@pytest.mark.asyncio
async def test_disconnect_unsubscribes_and_drops_future_publishes(monkeypatch) -> None:
    """After ``disconnect_socket`` runs, a subsequent SPUBLISH must NOT reach
    the dead websocket — exercises the SUNSUBSCRIBE + bookkeeping cleanup."""
    user_id = "user-conn-int-3"
    graph_id = f"graph-{uuid4().hex[:8]}"
    exec_id = f"exec-{uuid4().hex[:8]}"

    async def _fake_meta(_uid, gex_id):
        return _meta(user_id, graph_id, gex_id)

    monkeypatch.setattr("backend.api.conn_manager.get_graph_execution_meta", _fake_meta)

    cm = ConnectionManager()
    ws: AsyncMock = AsyncMock(spec=WebSocket)
    sent: list[str] = []
    ws.send_text = AsyncMock(side_effect=lambda m: sent.append(m))

    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    chan = event_bus_channel(exec_channel(user_id, graph_id, exec_id))
    payload = _node_event_payload(
        user_id=user_id, graph_id=graph_id, graph_exec_id=exec_id, marker="live"
    ).decode()

    try:
        await cm.subscribe_graph_exec(
            user_id=user_id, graph_exec_id=exec_id, websocket=ws
        )
        await asyncio.sleep(0.15)

        # First publish — must reach the socket.
        cluster.spublish(chan, payload)
        delivered = await _wait_until(lambda: bool(sent), timeout=5.0)
        assert delivered
        assert len(sent) == 1

        # Disconnect → SUNSUBSCRIBE + bookkeeping cleared.
        await cm.disconnect_socket(ws, user_id=user_id)
        # Pump cancellation may drain in-flight messages; wait for it.
        await asyncio.sleep(0.2)

        # Channel bookkeeping must be gone.
        assert (
            graph_exec_channel_key(user_id, graph_exec_id=exec_id)
            not in cm.subscriptions
        )
        assert ws not in cm._ws_subs

        # Second publish — must NOT reach the (already-disconnected) socket.
        cluster.spublish(
            chan,
            _node_event_payload(
                user_id=user_id,
                graph_id=graph_id,
                graph_exec_id=exec_id,
                marker="post-disconnect",
            ).decode(),
        )
        await asyncio.sleep(0.5)
        # Still only the one pre-disconnect message.
        assert len(sent) == 1
    finally:
        redis_client.disconnect()


@pytest.mark.asyncio
async def test_slow_consumer_receives_all_events_without_loss(monkeypatch) -> None:
    """Burst-publish many SPUBLISHes; assert every one reaches the subscriber
    in order — guards against drops/reorderings in the pubsub pump."""
    user_id = "user-conn-int-4"
    graph_id = f"graph-{uuid4().hex[:8]}"
    exec_id = f"exec-{uuid4().hex[:8]}"
    n_events = 100

    async def _fake_meta(_uid, gex_id):
        return _meta(user_id, graph_id, gex_id)

    monkeypatch.setattr("backend.api.conn_manager.get_graph_execution_meta", _fake_meta)

    cm = ConnectionManager()
    ws: AsyncMock = AsyncMock(spec=WebSocket)
    sent: list[str] = []
    ws.send_text = AsyncMock(side_effect=lambda m: sent.append(m))

    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    chan = event_bus_channel(exec_channel(user_id, graph_id, exec_id))

    try:
        await cm.subscribe_graph_exec(
            user_id=user_id, graph_exec_id=exec_id, websocket=ws
        )
        await asyncio.sleep(0.2)

        # Burst-publish n_events without yielding to the pump.
        for i in range(n_events):
            cluster.spublish(
                chan,
                _node_event_payload(
                    user_id=user_id,
                    graph_id=graph_id,
                    graph_exec_id=exec_id,
                    marker=f"m{i}",
                ).decode(),
            )

        delivered = await _wait_until(
            lambda: len(sent) >= n_events, timeout=15.0, interval=0.1
        )
        assert delivered, f"only delivered {len(sent)}/{n_events}"

        # Validate ordering — Redis pub/sub is FIFO per channel.
        markers = [json.loads(m)["data"]["input_data"]["in"] for m in sent[:n_events]]
        assert markers == [f"m{i}" for i in range(n_events)]
    finally:
        await cm.disconnect_socket(ws, user_id=user_id)
        redis_client.disconnect()
