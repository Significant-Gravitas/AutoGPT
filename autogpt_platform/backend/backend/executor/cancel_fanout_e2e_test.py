"""A graph-run cancel must reach every ExecutionManager pod, on a live broker.

Skips when no RabbitMQ is reachable; CI always runs one, on a per-shard vhost.
"""

from __future__ import annotations

import asyncio
import socket
from uuid import uuid4

import pytest

from backend.data.rabbitmq import SyncRabbitMQ, declare_broadcast_queue
from backend.executor.utils import (
    GRAPH_EXECUTION_CANCEL_EXCHANGE,
    CancelExecutionEvent,
    create_execution_queue_config,
)
from backend.util.settings import Settings


def _has_live_rabbit() -> bool:
    s = Settings()
    try:
        with socket.create_connection(
            (s.config.rabbitmq_host, s.config.rabbitmq_port), timeout=1.0
        ):
            return True
    except Exception:  # noqa: BLE001 - any connect failure → skip
        return False


rabbit_only = pytest.mark.skipif(
    not _has_live_rabbit(), reason="no RabbitMQ reachable; skip live-broker test"
)


@rabbit_only
async def test_a_cancel_reaches_every_pod_not_just_one() -> None:
    """Two pods, one cancel: both must see it.

    Each pod gets its own connection, as two ExecutionManager pods have.
    Sharing one queue instead makes the broker round-robin the cancel to
    exactly one of them, so the pod running the graph never stops.
    """
    graph_exec_id = str(uuid4())
    pods = [SyncRabbitMQ(create_execution_queue_config()) for _ in range(2)]
    queues: list[str] = []
    seen = [False, False]
    try:
        for pod in pods:
            pod.connect()
            queues.append(
                declare_broadcast_queue(
                    pod.get_channel(),
                    GRAPH_EXECUTION_CANCEL_EXCHANGE,
                    f"pod-{uuid4().hex[:6]}",
                )
            )

        publisher = pods[0]
        publisher.publish_message(
            routing_key="",
            message=CancelExecutionEvent(graph_exec_id=graph_exec_id).model_dump_json(),
            exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
        )

        for _ in range(50):
            seen = [
                got or _saw_cancel(pod, queue, graph_exec_id)
                for got, pod, queue in zip(seen, pods, queues)
            ]
            if all(seen):
                break
            await asyncio.sleep(0.1)
    finally:
        for pod in pods:
            pod.disconnect()

    assert seen == [True, True], (
        f"{seen.count(True)} of {len(pods)} pods received the cancel; "
        "a fanout bound to one shared queue delivers to exactly one consumer"
    )


def _saw_cancel(pod: SyncRabbitMQ, queue_name: str, graph_exec_id: str) -> bool:
    """Drain this pod's queue, reporting whether our cancel was among it."""
    channel = pod.get_channel()
    found = False
    while True:
        method, _properties, body = channel.basic_get(queue=queue_name, auto_ack=True)
        if method is None:
            return found
        event = CancelExecutionEvent.model_validate_json(body.decode())
        found = found or event.graph_exec_id == graph_exec_id
