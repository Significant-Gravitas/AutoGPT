"""Quorum-queue config assertions + mock-driven publish behaviour for
`AsyncRabbitMQ`. Live-broker scenarios live in `e2e_redis_rabbit_test.py`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aio_pika
import pytest

from backend.copilot.executor.utils import (
    COPILOT_EXECUTION_EXCHANGE,
    COPILOT_EXECUTION_QUEUE_NAME,
    COPILOT_EXECUTION_ROUTING_KEY,
    create_copilot_queue_config,
)
from backend.data.rabbitmq import (
    AsyncRabbitMQ,
    Exchange,
    ExchangeType,
    Queue,
    RabbitMQConfig,
    declare_broadcast_queue,
    reap_shared_queue,
)
from backend.executor.utils import (
    GRAPH_EXECUTION_CANCEL_EXCHANGE,
    GRAPH_EXECUTION_EXCHANGE,
    GRAPH_EXECUTION_QUEUE_NAME,
    GRAPH_EXECUTION_ROUTING_KEY,
    create_execution_queue_config,
)
from backend.notifications.queue import create_notification_config

# ---------- Quorum queue config: classic→quorum rollover guard ----------


@pytest.mark.parametrize(
    "factory",
    [
        create_execution_queue_config,
        create_copilot_queue_config,
        create_notification_config,
    ],
)
@pytest.mark.parametrize("vhost", ["/", "ci-shard"])
def test_queue_config_uses_configured_vhost(monkeypatch, factory, vhost: str) -> None:
    monkeypatch.setenv("RABBITMQ_VHOST", vhost)
    assert factory().vhost == vhost


def test_graph_execution_queue_is_quorum() -> None:
    """Run queue must declare `x-queue-type=quorum` to survive a single
    broker-node outage (AUTOGPT-SERVER-8ST/SV/SW)."""
    cfg = create_execution_queue_config()
    run = next(q for q in cfg.queues if q.name == GRAPH_EXECUTION_QUEUE_NAME)
    assert run.arguments is not None
    assert run.arguments.get("x-queue-type") == "quorum"
    # _v2 suffix marks the rollover so the old-image consumer keeps draining
    # the unsuffixed classic queue during a rolling deploy.
    assert run.name.endswith("_v2")
    assert run.durable is True
    assert run.exchange is GRAPH_EXECUTION_EXCHANGE


def test_graph_execution_config_declares_no_cancel_queue() -> None:
    """Cancels fan out to a per-pod exclusive queue each consumer declares
    itself; a queue here would be shared by the whole fleet again, and
    RabbitMQ would round-robin each cancel to one arbitrary pod."""
    cfg = create_execution_queue_config()
    assert GRAPH_EXECUTION_CANCEL_EXCHANGE in cfg.exchanges
    assert [q.name for q in cfg.queues] == [GRAPH_EXECUTION_QUEUE_NAME]


class TestDeclareBroadcastQueue:
    def test_queue_is_exclusive_auto_delete_and_bound_to_the_exchange(self) -> None:
        channel = MagicMock()
        name = declare_broadcast_queue(
            channel, GRAPH_EXECUTION_CANCEL_EXCHANGE, "executor-1"
        )
        assert name.startswith(f"{GRAPH_EXECUTION_CANCEL_EXCHANGE.name}.instance.")
        channel.queue_declare.assert_called_once_with(
            queue=name, durable=False, exclusive=True, auto_delete=True
        )
        channel.queue_bind.assert_called_once_with(
            queue=name, exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE.name, routing_key=""
        )

    def test_each_instance_gets_its_own_queue(self) -> None:
        names = {
            declare_broadcast_queue(
                MagicMock(), GRAPH_EXECUTION_CANCEL_EXCHANGE, instance_id
            )
            for instance_id in ("executor-1", "executor-2", "executor-1")
        }
        assert len(names) == 3


class TestReapSharedQueue:
    def test_leaves_a_queue_that_an_old_instance_still_drains(self) -> None:
        """Deleting it then would take that instance's broadcasts away."""
        channel = MagicMock()
        scratch = channel.connection.channel.return_value
        scratch.queue_declare.return_value.method.consumer_count = 1

        assert reap_shared_queue(channel, "old_queue") is False
        scratch.queue_delete.assert_not_called()
        scratch.close.assert_called_once()

    def test_deletes_a_queue_nothing_consumes(self) -> None:
        channel = MagicMock()
        scratch = channel.connection.channel.return_value
        scratch.queue_declare.return_value.method.consumer_count = 0

        assert reap_shared_queue(channel, "old_queue") is True
        scratch.queue_declare.assert_called_once_with(queue="old_queue", passive=True)
        scratch.queue_delete.assert_called_once_with(queue="old_queue")
        scratch.close.assert_called_once()

    def test_a_missing_queue_is_not_an_error(self) -> None:
        channel = MagicMock()
        scratch = channel.connection.channel.return_value
        scratch.queue_declare.side_effect = RuntimeError("NOT_FOUND")

        assert reap_shared_queue(channel, "old_queue") is False
        scratch.queue_delete.assert_not_called()


def test_copilot_execution_queue_is_quorum_with_consumer_timeout() -> None:
    """Copilot run queue must be quorum + carry a long consumer timeout
    matching the pod's graceful-shutdown window."""
    cfg = create_copilot_queue_config()
    run = next(q for q in cfg.queues if q.name == COPILOT_EXECUTION_QUEUE_NAME)
    assert run.arguments is not None
    assert run.arguments.get("x-queue-type") == "quorum"
    # Timeout must be in milliseconds and substantially larger than the
    # default 30-minute timeout so a 6-hour copilot turn doesn't get
    # cancelled by RabbitMQ mid-execution.
    timeout_ms = run.arguments.get("x-consumer-timeout")
    assert isinstance(timeout_ms, int)
    assert timeout_ms >= 60 * 60 * 1000  # at least 1 hour


def test_copilot_config_declares_no_cancel_queue() -> None:
    """Copilot cancels fan out to a per-pod exclusive queue the consumer
    declares itself; a queue here would be shared by the whole fleet again."""
    cfg = create_copilot_queue_config()
    assert [q.name for q in cfg.queues] == [COPILOT_EXECUTION_QUEUE_NAME]


# ---------- AsyncRabbitMQ.publish_message: mock-driven behaviour ----------


def _make_async_client(
    *, exchange_publish: AsyncMock | None = None
) -> tuple[AsyncRabbitMQ, MagicMock, MagicMock]:
    """Build an AsyncRabbitMQ wired to mock connection/channel/exchange.

    Returns the client, the mock channel, and the mock exchange so tests can
    assert on per-call arguments and tweak side_effects mid-flight.
    """
    cfg = RabbitMQConfig(
        vhost="/",
        exchanges=[
            Exchange(name="test_exchange", type=ExchangeType.DIRECT, durable=True)
        ],
        queues=[
            Queue(
                name="test_queue",
                exchange=Exchange(
                    name="test_exchange", type=ExchangeType.DIRECT, durable=True
                ),
                routing_key="rk",
                arguments={"x-queue-type": "quorum"},
            )
        ],
    )
    client = AsyncRabbitMQ(cfg)

    fake_exchange = MagicMock()
    fake_exchange.publish = exchange_publish or AsyncMock()

    fake_channel = MagicMock()
    fake_channel.is_closed = False
    fake_channel.get_exchange = AsyncMock(return_value=fake_exchange)
    fake_channel.default_exchange = fake_exchange

    fake_connection = MagicMock()
    fake_connection.is_closed = False

    client._connection = fake_connection
    client._channel = fake_channel
    return client, fake_channel, fake_exchange


@pytest.mark.asyncio
async def test_publish_100_messages_to_quorum_queue_all_confirmed() -> None:
    """A healthy quorum queue publish path must confirm 100/100 publishes
    with no NACKs."""
    client, _, fake_exchange = _make_async_client()
    exchange = Exchange(name="test_exchange", type=ExchangeType.DIRECT)
    for i in range(100):
        await client.publish_message(
            routing_key="rk", message=f"msg-{i}", exchange=exchange
        )
    assert fake_exchange.publish.await_count == 100
    # Every call carried a persistent message — durable on the broker side.
    for call in fake_exchange.publish.await_args_list:
        msg = call.args[0]
        assert isinstance(msg, aio_pika.Message)
        assert msg.delivery_mode == aio_pika.DeliveryMode.PERSISTENT


@pytest.mark.asyncio
async def test_publish_retries_on_delivery_error_then_raises() -> None:
    """Broker-side NACK (DeliveryError) must trigger ``func_retry`` and then
    raise gracefully if every retry fails — never crash the publisher loop."""
    publish = AsyncMock(
        side_effect=aio_pika.exceptions.DeliveryError(message=None, frame=None)
    )
    client, _, fake_exchange = _make_async_client(exchange_publish=publish)
    exchange = Exchange(name="test_exchange", type=ExchangeType.DIRECT)

    with pytest.raises(aio_pika.exceptions.DeliveryError):
        await client.publish_message(
            routing_key="rk", message="will-nack", exchange=exchange
        )
    # ``func_retry`` is configured for 5 attempts in retry.py — assert the
    # publisher attempted at least once but bounded retries.
    assert fake_exchange.publish.await_count >= 1
    assert fake_exchange.publish.await_count <= 10  # generous upper bound


@pytest.mark.asyncio
async def test_publish_retries_after_one_transient_failure() -> None:
    """A single transient DeliveryError must NOT propagate — ``func_retry``
    retries and the second call succeeds."""
    publish = AsyncMock(
        side_effect=[
            aio_pika.exceptions.DeliveryError(message=None, frame=None),
            None,  # second attempt succeeds
        ]
    )
    client, _, fake_exchange = _make_async_client(exchange_publish=publish)
    exchange = Exchange(name="test_exchange", type=ExchangeType.DIRECT)
    await client.publish_message(
        routing_key="rk", message="recovers", exchange=exchange
    )
    assert fake_exchange.publish.await_count == 2


@pytest.mark.asyncio
async def test_publish_reconnects_on_channel_invalid_state() -> None:
    """ChannelInvalidStateError must clear the channel and trigger a
    reconnect-and-retry — the publish_message wrapper handles this
    explicitly (see the except-clause in rabbitmq.py)."""
    publish = AsyncMock(
        side_effect=[
            aio_pika.exceptions.ChannelInvalidStateError("channel dead"),
            None,
        ]
    )
    client, fake_channel, fake_exchange = _make_async_client(exchange_publish=publish)
    exchange = Exchange(name="test_exchange", type=ExchangeType.DIRECT)

    # Patch connect() so the reconnect path doesn't try to hit a real broker.
    async def _fake_connect():
        # After reconnect the channel must be valid again.
        client._channel = fake_channel
        return None

    with patch.object(client, "connect", side_effect=_fake_connect):
        await client.publish_message(
            routing_key="rk", message="reconnects", exchange=exchange
        )
    # Two publish attempts: the failing one + the post-reconnect retry.
    assert fake_exchange.publish.await_count == 2


# ---------- Dual-deploy: legacy classic + new quorum publisher in parallel ----------


@pytest.mark.asyncio
async def test_dual_deploy_publishes_to_legacy_and_new_queues_in_parallel() -> None:
    """Rolling-deploy window: old-image producer publishes to classic queue,
    new-image to `_v2` quorum queue — both must succeed independently."""
    legacy_client, _, legacy_exchange = _make_async_client()
    new_client, _, new_exchange = _make_async_client()

    legacy_routing = "copilot.run"  # legacy producers used the same routing key
    new_routing = COPILOT_EXECUTION_ROUTING_KEY

    legacy_exch = Exchange(name="copilot_execution", type=ExchangeType.DIRECT)
    new_exch = Exchange(name=COPILOT_EXECUTION_EXCHANGE.name, type=ExchangeType.DIRECT)

    # Interleave 10 publishes from each producer — order doesn't matter.
    for i in range(10):
        await legacy_client.publish_message(
            routing_key=legacy_routing, message=f"legacy-{i}", exchange=legacy_exch
        )
        await new_client.publish_message(
            routing_key=new_routing, message=f"new-{i}", exchange=new_exch
        )

    assert legacy_exchange.publish.await_count == 10
    assert new_exchange.publish.await_count == 10

    # Each publisher's routing key landed on its own exchange — no crosstalk.
    for call in legacy_exchange.publish.await_args_list:
        assert call.kwargs.get("routing_key") == legacy_routing
    for call in new_exchange.publish.await_args_list:
        assert call.kwargs.get("routing_key") == new_routing


@pytest.mark.asyncio
async def test_dual_deploy_legacy_failure_does_not_affect_new_queue() -> None:
    """Legacy classic queue NACKing (AUTOGPT-SERVER-8ST) must not break
    publishes on the new `_v2` quorum queue."""
    legacy_publish = AsyncMock(
        side_effect=aio_pika.exceptions.DeliveryError(message=None, frame=None)
    )
    legacy_client, _, _ = _make_async_client(exchange_publish=legacy_publish)
    new_client, _, new_exchange = _make_async_client()

    legacy_exch = Exchange(name="copilot_execution", type=ExchangeType.DIRECT)
    new_exch = Exchange(name=COPILOT_EXECUTION_EXCHANGE.name, type=ExchangeType.DIRECT)

    # Legacy raises after retries — caller must catch it.
    with pytest.raises(aio_pika.exceptions.DeliveryError):
        await legacy_client.publish_message(
            routing_key="copilot.run", message="legacy-fail", exchange=legacy_exch
        )
    # New publisher continues to work — 5 successful publishes.
    for i in range(5):
        await new_client.publish_message(
            routing_key=COPILOT_EXECUTION_ROUTING_KEY,
            message=f"new-ok-{i}",
            exchange=new_exch,
        )
    assert new_exchange.publish.await_count == 5


# ---------- Configuration sanity for downstream queues ----------


def test_graph_execution_routing_key_constants() -> None:
    """Routing key + exchange wiring must stay aligned — guards against the
    classic→quorum migration accidentally also changing the routing key."""
    cfg = create_execution_queue_config()
    run = next(q for q in cfg.queues if q.name == GRAPH_EXECUTION_QUEUE_NAME)
    assert run.routing_key == GRAPH_EXECUTION_ROUTING_KEY
    assert GRAPH_EXECUTION_EXCHANGE in cfg.exchanges
