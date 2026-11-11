"""The route a run refused for want of a worker takes back to the run queue.

``create_execution_queue_config`` declares a queue nothing consumes, whose
dead-letter route returns a message to the run queue once the expiration the
executor publishes it with runs out. Those arguments are a payload the broker
executes, so a mocked channel accepts a typo in them without complaint while
the refused run stays parked.
"""

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pika
import pytest
from pika.adapters.blocking_connection import BlockingChannel

from backend.data.rabbitmq import Queue, RabbitMQConfig
from backend.executor.utils import (
    GRAPH_EXECUTION_DELAY_QUEUE_NAME,
    GRAPH_EXECUTION_DELAY_ROUTING_KEY,
    GRAPH_EXECUTION_EXCHANGE,
    GRAPH_EXECUTION_QUEUE_NAME,
    GRAPH_EXECUTION_ROUTING_KEY,
    create_execution_queue_config,
)
from backend.util.settings import Settings
from backend.util.testing import is_tcp_port_reachable

settings = Settings()
# Each in its own topology: a queue expires its head message only, so a parked
# one in front of an expiring one would hold it back. Long enough that reading
# an empty target straight after the publish is not a race on a loaded runner.
_PARKED_MS = "60000"
_EXPIRING_MS = "1000"


def test_the_delay_queue_dead_letters_onto_the_run_queue():
    config = create_execution_queue_config()
    delay = _queue(config, GRAPH_EXECUTION_DELAY_QUEUE_NAME)
    arguments = delay.arguments or {}

    assert arguments["x-dead-letter-exchange"] == GRAPH_EXECUTION_EXCHANGE.name
    assert arguments["x-dead-letter-routing-key"] == GRAPH_EXECUTION_ROUTING_KEY
    assert delay.routing_key == GRAPH_EXECUTION_DELAY_ROUTING_KEY
    assert (
        _queue(config, GRAPH_EXECUTION_QUEUE_NAME).routing_key
        == GRAPH_EXECUTION_ROUTING_KEY
    )


_needs_broker = pytest.mark.skipif(
    not is_tcp_port_reachable(
        settings.config.rabbitmq_host, settings.config.rabbitmq_port
    ),
    reason="needs a live RabbitMQ",
)


@_needs_broker
def test_the_broker_holds_a_published_message_for_its_expiration():
    with _throwaway_topology() as (channel, exchange, delay_queue, target_queue):
        _publish(channel, exchange, b'{"parked": true}', _PARKED_MS)

        assert channel.basic_get(target_queue, auto_ack=True)[0] is None
        assert _message_count(channel, delay_queue) == 1


@_needs_broker
def test_the_broker_routes_an_expired_message_on():
    with _throwaway_topology() as (channel, exchange, _, target_queue):
        _publish(channel, exchange, b'{"expiring": true}', _EXPIRING_MS)

        assert _await_message(channel, target_queue) == b'{"expiring": true}'


@contextmanager
def _throwaway_topology() -> Iterator[tuple[BlockingChannel, str, str, str]]:
    """The delay queue's own arguments, under names this test owns.

    Only the route out is redirected, so nothing here can reach the run queue;
    that the configured route names the run queue is asserted above.
    """
    arguments = dict(
        _queue(
            create_execution_queue_config(), GRAPH_EXECUTION_DELAY_QUEUE_NAME
        ).arguments
        or {}
    )
    tag = uuid.uuid4().hex[:8]
    exchange = f"delay_queue_test_{tag}"
    delay_queue, target_queue = f"{exchange}_delay", f"{exchange}_target"
    arguments["x-dead-letter-exchange"] = exchange

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=settings.config.rabbitmq_host,
            port=settings.config.rabbitmq_port,
            virtual_host=settings.config.rabbitmq_vhost,
            credentials=pika.PlainCredentials(
                settings.secrets.rabbitmq_default_user,
                settings.secrets.rabbitmq_default_pass,
            ),
            socket_timeout=10,
        )
    )
    try:
        channel = connection.channel()
        # So an unroutable publish raises instead of dropping the message.
        channel.confirm_delivery()
        channel.exchange_declare(
            exchange, exchange_type="direct", durable=True, auto_delete=False
        )
        channel.queue_declare(target_queue, durable=True)
        channel.queue_bind(target_queue, exchange, GRAPH_EXECUTION_ROUTING_KEY)
        channel.queue_declare(delay_queue, durable=True, arguments=arguments)
        channel.queue_bind(delay_queue, exchange, GRAPH_EXECUTION_DELAY_ROUTING_KEY)
        yield channel, exchange, delay_queue, target_queue
    finally:
        cleanup = connection.channel()
        for queue_name in (delay_queue, target_queue):
            cleanup.queue_delete(queue_name)
        cleanup.exchange_delete(exchange)
        connection.close()


def _publish(
    channel: BlockingChannel, exchange: str, body: bytes, expiration: str
) -> None:
    channel.basic_publish(
        exchange=exchange,
        routing_key=GRAPH_EXECUTION_DELAY_ROUTING_KEY,
        body=body,
        properties=pika.BasicProperties(delivery_mode=2, expiration=expiration),
        mandatory=True,
    )


def _await_message(
    channel: BlockingChannel, queue_name: str, timeout: float = 30.0
) -> bytes | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, _, body = channel.basic_get(queue_name, auto_ack=True)
        if body is not None:
            return body
        time.sleep(0.2)
    return None


def _message_count(channel: BlockingChannel, queue_name: str) -> int:
    return channel.queue_declare(queue_name, passive=True).method.message_count


def _queue(config: RabbitMQConfig, name: str) -> Queue:
    (queue,) = [queue for queue in config.queues if queue.name == name]
    return queue
