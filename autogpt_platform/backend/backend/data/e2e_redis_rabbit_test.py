"""End-to-end coverage of the data-layer APIs over the live 3-shard Redis
cluster + RabbitMQ broker. Tests skip when their infra is unreachable.
Container-restart scenarios live in `e2e_redis_restart_test.py`."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest

import backend.data.redis_client as redis_client
from backend.api.model import NotificationPayload
from backend.data.execution import (
    AsyncRedisExecutionEventBus,
    ExecutionStatus,
    NodeExecutionEvent,
    exec_channel,
    graph_all_channel,
)
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)
from backend.data.rabbitmq import AsyncRabbitMQ
from backend.executor.utils import (
    GRAPH_EXECUTION_EXCHANGE,
    GRAPH_EXECUTION_QUEUE_NAME,
    create_execution_queue_config,
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


def _has_live_rabbit() -> bool:
    """Probe the rabbitmq host:port from settings; skip if unreachable."""
    import socket

    from backend.util.settings import Settings

    s = Settings()
    try:
        with socket.create_connection(
            (s.config.rabbitmq_host, s.config.rabbitmq_port), timeout=1.0
        ):
            return True
    except Exception:  # noqa: BLE001 - any connect failure → skip the test
        return False


cluster_only = pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip e2e integration",
)
rabbit_only = pytest.mark.skipif(
    not _has_live_rabbit(),
    reason="local rabbitmq not reachable; skip e2e integration",
)


def _make_node_event(*, user_id: str, graph_id: str, gex_id: str, marker: str):
    return NodeExecutionEvent(
        user_id=user_id,
        graph_id=graph_id,
        graph_version=1,
        graph_exec_id=gex_id,
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
    )


# ---------- Scenario 1: cluster cache round-trip across slots ----------


@cluster_only
def test_cluster_cache_roundtrip_across_three_slots() -> None:
    """A list-graphs-style cache flow: SET keys with hash tags that land on
    different shards, GET them back. Validates the basic cluster-routing
    contract end-to-end."""
    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    keys = []
    try:
        # Pick keys that hash to different slots — try until 3 distinct shards.
        seen: set[tuple[str, int]] = set()
        for i in range(2000):
            key = f"e2e:cache:{i}"
            node = cluster.get_node_from_key(key)
            owner = (node.host, node.port)
            if owner in seen:
                continue
            seen.add(owner)
            keys.append(key)
            if len(seen) >= 3:
                break
        assert len(keys) >= 3

        for i, k in enumerate(keys):
            cluster.setex(k, 60, f"v-{i}")
        for i, k in enumerate(keys):
            assert cluster.get(k) == f"v-{i}"
    finally:
        for k in keys:
            try:
                cluster.delete(k)
            except Exception:
                pass
        redis_client.disconnect()


# ---------- Scenarios 2 & 3: graph execution event streams ----------


@pytest.mark.asyncio
@cluster_only
async def test_graph_execution_events_complete_under_ten_seconds() -> None:
    """A listener subscribes to the per-exec channel; the producer publishes
    one node event. The listener must observe it in under 10 seconds —
    pins the latency contract end-to-end through SPUBLISH/SSUBSCRIBE."""
    redis_client._async_clients.clear()
    user_id = f"u-e2e-{uuid4().hex[:8]}"
    graph_id = f"g-{uuid4().hex[:8]}"
    gex_id = f"x-{uuid4().hex[:8]}"

    publisher = AsyncRedisExecutionEventBus()
    subscriber = AsyncRedisExecutionEventBus()
    received: list[str] = []

    async def _consume() -> None:
        async for evt in subscriber.listen_events(
            exec_channel(user_id, graph_id, gex_id)
        ):
            received.append(getattr(evt, "node_exec_id", "graph"))
            return

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.3)

    start = time.monotonic()
    try:
        await publisher.publish_event(
            _make_node_event(
                user_id=user_id, graph_id=graph_id, gex_id=gex_id, marker="m1"
            ),
            exec_channel(user_id, graph_id, gex_id),
        )
        await asyncio.wait_for(task, timeout=10.0)
    finally:
        if not task.done():
            task.cancel()
        await subscriber.close()
        await redis_client.disconnect_async()

    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"event roundtrip took {elapsed:.2f}s, expected < 10s"
    assert received == ["node-exec-m1"]


@pytest.mark.asyncio
@cluster_only
async def test_two_concurrent_graphs_no_cross_talk() -> None:
    """Two graphs execute in parallel; two listeners on different per-exec
    channels each receive only their own events."""
    redis_client._async_clients.clear()
    user_id = f"u-e2e-{uuid4().hex[:8]}"
    g1, g2 = f"g1-{uuid4().hex[:8]}", f"g2-{uuid4().hex[:8]}"
    e1, e2 = f"e1-{uuid4().hex[:8]}", f"e2-{uuid4().hex[:8]}"

    publisher = AsyncRedisExecutionEventBus()
    sub_a = AsyncRedisExecutionEventBus()
    sub_b = AsyncRedisExecutionEventBus()

    async def _listen_one(bus, channel_key: str, sink: list, want: int) -> None:
        async for evt in bus.listen_events(channel_key):
            sink.append(getattr(evt, "node_exec_id", "graph"))
            if len(sink) >= want:
                return

    sink_a: list[str] = []
    sink_b: list[str] = []
    t_a = asyncio.create_task(
        _listen_one(sub_a, exec_channel(user_id, g1, e1), sink_a, want=3)
    )
    t_b = asyncio.create_task(
        _listen_one(sub_b, exec_channel(user_id, g2, e2), sink_b, want=3)
    )
    await asyncio.sleep(0.3)

    try:
        for i in range(3):
            await publisher.publish_event(
                _make_node_event(
                    user_id=user_id, graph_id=g1, gex_id=e1, marker=f"a{i}"
                ),
                exec_channel(user_id, g1, e1),
            )
            await publisher.publish_event(
                _make_node_event(
                    user_id=user_id, graph_id=g2, gex_id=e2, marker=f"b{i}"
                ),
                exec_channel(user_id, g2, e2),
            )

        await asyncio.wait_for(asyncio.gather(t_a, t_b), timeout=10.0)
        assert sink_a == ["node-exec-a0", "node-exec-a1", "node-exec-a2"]
        assert sink_b == ["node-exec-b0", "node-exec-b1", "node-exec-b2"]
    finally:
        await sub_a.close()
        await sub_b.close()
        await redis_client.disconnect_async()


# ---------- Scenario 4: aggregate /all channel for graph executions ----------


@pytest.mark.asyncio
@cluster_only
async def test_three_executions_land_on_aggregate_channel() -> None:
    """Subscribe to the aggregate ``/all`` channel; trigger 3 different
    executions of the same graph; assert all 3 land on the aggregate."""
    redis_client._async_clients.clear()
    user_id = f"u-e2e-{uuid4().hex[:8]}"
    graph_id = f"g-{uuid4().hex[:8]}"
    exec_ids = [f"x{i}-{uuid4().hex[:6]}" for i in range(3)]

    publisher = AsyncRedisExecutionEventBus()
    subscriber = AsyncRedisExecutionEventBus()
    received: list[str] = []

    async def _listen_all() -> None:
        async for evt in subscriber.listen_events(graph_all_channel(user_id, graph_id)):
            received.append(getattr(evt, "graph_exec_id", "?"))
            if len(received) >= 3:
                return

    task = asyncio.create_task(_listen_all())
    await asyncio.sleep(0.3)

    try:
        for ex in exec_ids:
            await publisher.publish_event(
                _make_node_event(
                    user_id=user_id, graph_id=graph_id, gex_id=ex, marker=ex
                ),
                graph_all_channel(user_id, graph_id),
            )

        await asyncio.wait_for(task, timeout=10.0)
        # Order of receipt may vary slightly under load — check set membership.
        assert set(received) == set(exec_ids)
    finally:
        await subscriber.close()
        await redis_client.disconnect_async()


# ---------- Scenarios 5 & 6: copilot/notification per-user channels ----------


@pytest.mark.asyncio
@cluster_only
async def test_copilot_cancel_signal_via_sharded_pubsub() -> None:
    """A subscriber on a per-session channel receives an SPUBLISH cancel
    signal — the primitive the copilot executor uses for graceful cancel."""
    redis_client._async_clients.clear()
    session_id = f"sess-{uuid4().hex[:8]}"
    channel = "{copilot/" + session_id + "}/cancel"

    client = await redis_client.connect_sharded_pubsub_async(channel)
    pubsub = client.pubsub()
    received: list[str] = []
    try:
        await pubsub.execute_command("SSUBSCRIBE", channel)
        # Prime the channels map so listen() doesn't early-exit (see _Subscription).
        pubsub.channels[channel] = None  # type: ignore[index]

        async def _pump() -> None:
            async for msg in pubsub.listen():
                if msg.get("type") == "smessage":
                    received.append(msg["data"])
                    return

        listener = asyncio.create_task(_pump())
        await asyncio.sleep(0.2)

        cluster = await redis_client.get_redis_async()
        await cluster.execute_command("SPUBLISH", channel, "cancel")

        await asyncio.wait_for(listener, timeout=5.0)
        assert received == ["cancel"]
    finally:
        try:
            await pubsub.execute_command("SUNSUBSCRIBE", channel)
        except Exception:
            pass
        await pubsub.aclose()
        await client.aclose()
        await redis_client.disconnect_async()


@pytest.mark.asyncio
@cluster_only
async def test_notification_fan_out_per_user_channel() -> None:
    """Per-user SSUBSCRIBE: a publish on the user's notification channel
    reaches the user's listener and only that listener."""
    redis_client._async_clients.clear()
    user_id = f"u-notif-{uuid4().hex[:8]}"
    other_user_id = f"u-other-{uuid4().hex[:8]}"

    publisher = AsyncRedisNotificationEventBus()
    listener_user = AsyncRedisNotificationEventBus()
    listener_other = AsyncRedisNotificationEventBus()

    user_received: list[str] = []
    other_received: list[str] = []
    notif_for_user = NotificationEvent(
        user_id=user_id,
        payload=NotificationPayload(type="info", event="balance-low"),
    )
    notif_for_other = NotificationEvent(
        user_id=other_user_id,
        payload=NotificationPayload(type="info", event="other"),
    )

    async def _listen_one(bus: AsyncRedisNotificationEventBus, uid: str, sink: list):
        async for evt in bus.listen(uid):
            sink.append(evt.user_id)
            return

    t_user = asyncio.create_task(_listen_one(listener_user, user_id, user_received))
    t_other = asyncio.create_task(
        _listen_one(listener_other, other_user_id, other_received)
    )
    await asyncio.sleep(0.3)

    try:
        await publisher.publish(notif_for_user)
        await publisher.publish(notif_for_other)
        await asyncio.wait_for(asyncio.gather(t_user, t_other), timeout=10.0)
        assert user_received == [user_id]
        assert other_received == [other_user_id]
    finally:
        await listener_user.close()
        await listener_other.close()
        await publisher.close()
        await redis_client.disconnect_async()


# ---------- Scenario 7: idle WS connection 60s ----------


@pytest.mark.asyncio
@cluster_only
async def test_idle_subscriber_60s_then_receives_publish() -> None:
    """An SSUBSCRIBE that sits idle past one health-check interval must
    still deliver a subsequent SPUBLISH (uses HEALTH_CHECK_INTERVAL+5s)."""
    redis_client._async_clients.clear()
    channel = "{idle-e2e}/exec/" + uuid4().hex[:8]
    client = await redis_client.connect_sharded_pubsub_async(channel)
    pubsub = client.pubsub()
    try:
        await pubsub.execute_command("SSUBSCRIBE", channel)
        pubsub.channels[channel] = None  # type: ignore[index]
        # Drain ssubscribe confirm.
        async for _msg in pubsub.listen():
            break

        idle_seconds = redis_client.HEALTH_CHECK_INTERVAL + 5
        await asyncio.sleep(idle_seconds)

        cluster = await redis_client.get_redis_async()
        await cluster.execute_command("SPUBLISH", channel, "hello-after-idle")

        async for msg in pubsub.listen():
            if msg.get("type") == "smessage":
                assert msg["data"] == "hello-after-idle"
                return
    finally:
        try:
            await pubsub.execute_command("SUNSUBSCRIBE", channel)
        except Exception:
            pass
        await pubsub.aclose()
        await client.aclose()
        await redis_client.disconnect_async()


# ---------- Scenario 8: graph_execution_queue_v2 publish + consume ----------


@pytest.mark.asyncio
@rabbit_only
async def test_graph_execution_queue_publish_and_consume() -> None:
    """End-to-end on a test-scoped quorum queue: publish via AsyncRabbitMQ
    → consume → payload round-trips intact. Uses a unique routing key so
    the live executor consumer (if any) doesn't race for the message."""
    from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig

    test_queue_name = f"e2e_test_{uuid4().hex[:8]}_v2"
    test_routing_key = f"e2e.test.{uuid4().hex[:8]}"
    test_exchange = Exchange(
        name=GRAPH_EXECUTION_EXCHANGE.name,
        type=ExchangeType.DIRECT,
        durable=True,
    )
    test_queue = Queue(
        name=test_queue_name,
        durable=True,
        # Quorum queues reject auto_delete; we delete the queue explicitly
        # in the finally block instead.
        auto_delete=False,
        exchange=test_exchange,
        routing_key=test_routing_key,
        arguments={"x-queue-type": "quorum"},
    )
    cfg = RabbitMQConfig(vhost="/", exchanges=[test_exchange], queues=[test_queue])

    publisher = AsyncRabbitMQ(cfg)
    await publisher.connect()
    consumer = AsyncRabbitMQ(cfg)
    await consumer.connect()

    payload = json.dumps(
        {"graph_exec_id": f"e2e-{uuid4().hex[:8]}", "marker": "round-trip"}
    )

    try:
        channel = await consumer.get_channel()
        queue_obj = await channel.get_queue(test_queue_name)

        await publisher.publish_message(
            routing_key=test_routing_key,
            message=payload,
            exchange=test_exchange,
        )

        # Poll get() — quorum queue must surface the publish within 5s.
        deadline = time.monotonic() + 5.0
        msg = None
        while time.monotonic() < deadline:
            msg = await queue_obj.get(no_ack=True, fail=False)
            if msg is not None:
                break
            await asyncio.sleep(0.05)
        assert msg is not None, "publish never reached the quorum queue"
        assert msg.body.decode() == payload
    finally:
        # Best-effort delete in case auto_delete didn't trigger.
        try:
            channel = await consumer.get_channel()
            await channel.queue_delete(test_queue_name, if_unused=False, if_empty=False)
        except Exception:
            pass
        await publisher.disconnect()
        await consumer.disconnect()


@pytest.mark.asyncio
@rabbit_only
async def test_graph_execution_queue_uses_quorum_via_real_broker() -> None:
    """Live-broker check that `graph_execution_queue_v2` is declared as
    quorum — passive re-declare with `x-queue-type=quorum` must not raise."""
    cfg = create_execution_queue_config()
    client = AsyncRabbitMQ(cfg)
    await client.connect()  # declares everything in cfg
    try:
        channel = await client.get_channel()
        # Re-declare passively — must NOT raise PRECONDITION_FAILED if the
        # type matches, would raise if quorum was lost.
        q = await channel.declare_queue(
            name=GRAPH_EXECUTION_QUEUE_NAME,
            durable=True,
            arguments={"x-queue-type": "quorum"},
            passive=True,
        )
        assert q.name == GRAPH_EXECUTION_QUEUE_NAME
    finally:
        await client.disconnect()
