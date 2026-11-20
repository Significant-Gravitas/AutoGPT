"""The cancel fanout must reach every executor pod, against a live broker.

Skips when no RabbitMQ is reachable; CI always runs one, on a per-shard vhost.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import suppress
from uuid import uuid4

import pytest

from backend.copilot.executor import utils
from backend.copilot.executor.utils import (
    COPILOT_CANCEL_EXCHANGE,
    CancelCoPilotEvent,
    create_copilot_queue_config,
    declare_pod_cancel_queue,
    enqueue_cancel_task,
    reap_legacy_cancel_queue,
)
from backend.data.rabbitmq import SyncRabbitMQ
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

    Each pod gets its own connection, as two executor pods have. Sharing one
    queue instead makes the broker round-robin the cancel to exactly one of
    them, so the pod holding the session never stops.
    """
    session_id = str(uuid4())
    pods = [SyncRabbitMQ(create_copilot_queue_config()) for _ in range(2)]
    queues: list[str] = []
    seen = [False, False]
    try:
        for pod in pods:
            pod.connect()
            queues.append(
                declare_pod_cancel_queue(pod.get_channel(), f"pod-{uuid4().hex[:6]}")
            )

        await enqueue_cancel_task(session_id)

        for _ in range(50):
            seen = [
                got or _saw_cancel(pod, queue, session_id)
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


@rabbit_only
async def test_the_retired_queue_is_reaped_only_once_nothing_drains_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retired fleet-wide queue goes away on its own, with no operator step.

    Deleting it while an old-image pod is still draining it would take that
    pod's cancels away, so the consumer count is the gate. Runs against a
    scratch name: the reaper deletes whatever ``LEGACY_...`` points at.
    """
    legacy = f"copilot_cancel_queue_v2_test_{uuid4().hex[:8]}"
    monkeypatch.setattr(utils, "LEGACY_COPILOT_CANCEL_QUEUE_NAME", legacy)

    new_pod = SyncRabbitMQ(create_copilot_queue_config())
    old_pod = SyncRabbitMQ(create_copilot_queue_config())
    try:
        new_pod.connect()
        old_pod.connect()
        declare_pod_cancel_queue(new_pod.get_channel(), "new-pod")

        old_channel = old_pod.get_channel()
        old_channel.queue_declare(
            queue=legacy, durable=True, arguments={"x-queue-type": "quorum"}
        )
        # bound the way declare_infrastructure binds it: `routing_key or name`
        old_channel.queue_bind(
            queue=legacy, exchange=COPILOT_CANCEL_EXCHANGE.name, routing_key=legacy
        )
        old_channel.basic_consume(
            queue=legacy, on_message_callback=lambda *_: None, auto_ack=True
        )
        old_channel.connection.process_data_events(time_limit=1)

        assert reap_legacy_cancel_queue(new_pod.get_channel()) is False
        assert _queue_exists(new_pod, legacy), "reaped a queue an old pod still drains"

        old_pod.disconnect()  # the last old-image pod finishes draining and goes

        # the broker drops the consumer count asynchronously, and this reads it
        # over a different connection, so poll rather than assert on one pass
        reaped = False
        for _ in range(50):
            reaped = reap_legacy_cancel_queue(new_pod.get_channel())
            if reaped:
                break
            await asyncio.sleep(0.1)

        assert reaped, "the retired queue outlived its last consumer"
        assert not _queue_exists(new_pod, legacy)
    finally:
        if new_pod.is_ready:  # a failed run must not leave it bound to the fanout
            with suppress(Exception):
                new_pod.get_channel().queue_delete(queue=legacy)
        for pod in (old_pod, new_pod):
            pod.disconnect()


def _queue_exists(pod: SyncRabbitMQ, queue_name: str) -> bool:
    """Ask the broker, on a scratch channel: a 404 closes the channel it hits."""
    scratch = pod.get_channel().connection.channel()
    try:
        scratch.queue_declare(queue=queue_name, passive=True)
        return True
    except Exception:  # noqa: BLE001 - 404 is the answer we are after
        return False
    finally:
        if scratch.is_open:
            scratch.close()


def _saw_cancel(pod: SyncRabbitMQ, queue_name: str, session_id: str) -> bool:
    """Drain this pod's queue, reporting whether our cancel was among it."""
    channel = pod.get_channel()
    found = False
    while True:
        method, _properties, body = channel.basic_get(queue=queue_name, auto_ack=True)
        if method is None:
            return found
        event = CancelCoPilotEvent.model_validate_json(body.decode())
        found = found or event.session_id == session_id
