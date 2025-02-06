import logging
import os
from typing import Optional, Callable, Any

from dotenv import load_dotenv
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractConnection,
    AbstractQueue,
    AbstractIncomingMessage,
)

from backend.util.retry import conn_retry

load_dotenv()

HOST = os.getenv("RABBITMQ_HOST", "localhost")
PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
USERNAME = os.getenv("RABBITMQ_DEFAULT_USER", "guest")
PASSWORD = os.getenv("RABBITMQ_DEFAULT_PASS", "guest")
VIRTUAL_HOST = os.getenv("RABBITMQ_VHOST", "/")

logger = logging.getLogger(__name__)

# Synchronous connections
connection: Optional[pika.BlockingConnection] = None
channel: Optional[BlockingChannel] = None

# Async connections
connection_async: Optional[AbstractConnection] = None
channel_async: Optional[AbstractChannel] = None


@conn_retry("RabbitMQ", "Acquiring connection")
def connect() -> pika.BlockingConnection:
    global connection, channel
    if connection and connection.is_open:
        return connection

    credentials = pika.PlainCredentials(USERNAME, PASSWORD)
    parameters = pika.ConnectionParameters(
        host=HOST,
        port=PORT,
        virtual_host=VIRTUAL_HOST,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300,
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)
    return connection


@conn_retry("RabbitMQ", "Releasing connection")
def disconnect():
    global connection, channel
    if channel:
        if channel.is_open:
            channel.close()
        channel = None
    if connection:
        if connection.is_open:
            connection.close()
        connection = None


def get_channel(auto_connect: bool = True) -> BlockingChannel:
    global channel
    if channel and channel.is_open:
        return channel
    if auto_connect:
        connect()
        # This may seem like a wierd place to double check, but its a good sanity check
        # and fixes the type checker since we are relying on globals
        if not channel:
            raise RuntimeError("RabbitMQ connection is not established")
        return channel
    raise RuntimeError("RabbitMQ connection is not established")


def declare_queue(
    queue_name: str, durable: bool = True, auto_delete: bool = False
) -> None:
    ch = get_channel()
    ch.queue_declare(queue=queue_name, durable=durable, auto_delete=auto_delete)


def publish_message(
    queue_name: str,
    message: str,
    properties: Optional[BasicProperties] = None,
    mandatory: bool = True,
) -> None:
    ch = get_channel()
    ch.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=message.encode(),
        properties=properties or BasicProperties(delivery_mode=2),
        mandatory=mandatory,
    )


def consume_messages(
    queue_name: str,
    callback: Callable[[BlockingChannel, Basic.Deliver, BasicProperties, str], Any],
    auto_ack: bool = False,
) -> None:
    ch = get_channel()
    ch.basic_consume(
        queue=queue_name,
        on_message_callback=lambda ch, method, props, body: callback(
            ch, method, props, body.decode()
        ),
        auto_ack=auto_ack,
    )
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        ch.stop_consuming()


# Async Section
@conn_retry("AsyncRabbitMQ", "Acquiring async connection")
async def connect_async() -> AbstractConnection:
    global connection_async, channel_async
    if connection_async and not connection_async.is_closed:
        return connection_async

    connection_async = await aio_pika.connect_robust(
        host=HOST,
        port=PORT,
        login=USERNAME,
        password=PASSWORD,
        virtualhost=VIRTUAL_HOST,
    )
    channel_async = await connection_async.channel()
    await channel_async.set_qos(prefetch_count=1)
    return connection_async


@conn_retry("AsyncRabbitMQ", "Releasing async connection")
async def disconnect_async():
    global connection_async, channel_async
    if channel_async:
        await channel_async.close()
        channel_async = None
    if connection_async:
        await connection_async.close()
        connection_async = None


async def get_channel_async(auto_connect: bool = True) -> AbstractChannel:
    global channel_async
    if channel_async and not channel_async.is_closed:
        return channel_async
    if auto_connect:
        await connect_async()
        # This may seem like a wierd place to double check, but its a good sanity check
        # and fixes the type checker since we are relying on globals
        if not channel_async:
            raise RuntimeError("AsyncRabbitMQ connection is not established")
        return channel_async
    raise RuntimeError("AsyncRabbitMQ connection is not established")


async def declare_queue_async(
    queue_name: str, durable: bool = True, auto_delete: bool = False
) -> AbstractQueue:
    ch = await get_channel_async()
    return await ch.declare_queue(
        name=queue_name, durable=durable, auto_delete=auto_delete
    )


async def publish_message_async(
    queue_name: str, message: str, persistent: bool = True
) -> None:
    ch = await get_channel_async()
    await ch.default_exchange.publish(
        aio_pika.Message(
            body=message.encode(),
            delivery_mode=(
                aio_pika.DeliveryMode.PERSISTENT
                if persistent
                else aio_pika.DeliveryMode.NOT_PERSISTENT
            ),
        ),
        routing_key=queue_name,
    )


async def consume_messages_async(
    queue_name: str,
    callback: Callable[[AbstractIncomingMessage], Any],
    auto_ack: bool = False,
) -> None:
    queue = await declare_queue_async(queue_name)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            try:
                if auto_ack:
                    await message.ack()
                await callback(message)
                if not auto_ack:
                    await message.ack()
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
                await message.reject(requeue=True)


# Example usage of callback functions
def sync_message_handler(
    channel: BlockingChannel,
    method: Basic.Deliver,
    properties: BasicProperties,
    body: str,
) -> None:
    logger.info(f"Received message: {body}")
    channel.basic_ack(delivery_tag=method.delivery_tag)


async def async_message_handler(message: AbstractIncomingMessage) -> None:
    async with message.process():
        logger.info(f"Received message: {message.body.decode()}")
