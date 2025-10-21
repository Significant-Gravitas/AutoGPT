"""
In-memory queue implementation as a fallback for RabbitMQ.
This allows running the platform without an external RabbitMQ server.
"""
import asyncio
import logging
from queue import Queue, Empty
from threading import Lock
from typing import Optional, Callable, Awaitable
from collections import defaultdict

from backend.data.rabbitmq import (
    RabbitMQBase,
    RabbitMQConfig,
    Exchange,
    Queue as QueueConfig,
)

logger = logging.getLogger(__name__)


class InMemoryMessageBroker:
    """
    Singleton in-memory message broker that manages all queues and exchanges.
    Thread-safe implementation.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._queues = {}
                    cls._instance._async_queues = {}
                    cls._instance._exchanges = {}
                    cls._instance._bindings = defaultdict(list)  # exchange -> [(queue, routing_key)]
                    cls._instance._consumers = {}  # queue -> [callbacks]
                    cls._instance._async_consumers = {}  # queue -> [async callbacks]
        return cls._instance

    def declare_exchange(self, exchange: Exchange):
        """Declare an exchange"""
        with self._lock:
            if exchange.name not in self._exchanges:
                self._exchanges[exchange.name] = exchange
                logger.debug(f"Declared exchange: {exchange.name} (type: {exchange.type})")

    def declare_queue(self, queue: QueueConfig):
        """Declare a queue"""
        with self._lock:
            if queue.name not in self._queues:
                self._queues[queue.name] = Queue()
                self._async_queues[queue.name] = asyncio.Queue()
                logger.debug(f"Declared queue: {queue.name}")

    def bind_queue(self, queue_name: str, exchange_name: str, routing_key: str):
        """Bind a queue to an exchange with a routing key"""
        with self._lock:
            self._bindings[exchange_name].append((queue_name, routing_key))
            logger.debug(f"Bound queue {queue_name} to exchange {exchange_name} with key {routing_key}")

    def publish(self, exchange_name: str, routing_key: str, message: str):
        """Publish a message to an exchange"""
        with self._lock:
            if not exchange_name:
                # Direct publish to queue (default exchange)
                if routing_key in self._queues:
                    self._queues[routing_key].put(message)
                    logger.debug(f"Published to queue {routing_key}: {message[:100]}...")
                return

            # Publish to exchange
            exchange = self._exchanges.get(exchange_name)
            if not exchange:
                logger.warning(f"Exchange {exchange_name} not found")
                return

            # Route message based on exchange type
            for queue_name, bound_key in self._bindings.get(exchange_name, []):
                if self._matches_routing_key(exchange, routing_key, bound_key):
                    if queue_name in self._queues:
                        self._queues[queue_name].put(message)
                        logger.debug(f"Routed to queue {queue_name}: {message[:100]}...")

    async def publish_async(self, exchange_name: str, routing_key: str, message: str):
        """Async version of publish"""
        with self._lock:
            if not exchange_name:
                # Direct publish to queue (default exchange)
                if routing_key in self._async_queues:
                    await self._async_queues[routing_key].put(message)
                    logger.debug(f"Published async to queue {routing_key}: {message[:100]}...")
                return

            # Publish to exchange
            exchange = self._exchanges.get(exchange_name)
            if not exchange:
                logger.warning(f"Exchange {exchange_name} not found")
                return

            # Route message based on exchange type
            for queue_name, bound_key in self._bindings.get(exchange_name, []):
                if self._matches_routing_key(exchange, routing_key, bound_key):
                    if queue_name in self._async_queues:
                        await self._async_queues[queue_name].put(message)
                        logger.debug(f"Routed async to queue {queue_name}: {message[:100]}...")

    def get_message(self, queue_name: str, timeout: float = 1.0) -> Optional[str]:
        """Get a message from a queue"""
        if queue_name not in self._queues:
            return None
        try:
            return self._queues[queue_name].get(timeout=timeout)
        except Empty:
            return None

    async def get_message_async(self, queue_name: str, timeout: float = 1.0) -> Optional[str]:
        """Async version of get_message"""
        if queue_name not in self._async_queues:
            return None
        try:
            return await asyncio.wait_for(
                self._async_queues[queue_name].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    @staticmethod
    def _matches_routing_key(exchange: Exchange, routing_key: str, bound_key: str) -> bool:
        """Check if a routing key matches a bound key based on exchange type"""
        from backend.data.rabbitmq import ExchangeType

        if exchange.type == ExchangeType.FANOUT:
            # Fanout exchanges ignore routing keys
            return True
        elif exchange.type == ExchangeType.DIRECT:
            # Direct exchanges require exact match
            return routing_key == bound_key
        elif exchange.type == ExchangeType.TOPIC:
            # Topic exchanges support wildcards (* and #)
            return InMemoryMessageBroker._match_topic(routing_key, bound_key)
        else:
            # Headers exchange not implemented, default to exact match
            return routing_key == bound_key

    @staticmethod
    def _match_topic(routing_key: str, pattern: str) -> bool:
        """Match a routing key against a topic pattern"""
        routing_parts = routing_key.split('.')
        pattern_parts = pattern.split('.')

        if len(pattern_parts) > len(routing_parts) and '#' not in pattern:
            return False

        for i, pattern_part in enumerate(pattern_parts):
            if pattern_part == '#':
                # # matches zero or more words
                return True
            if pattern_part == '*':
                # * matches exactly one word
                if i >= len(routing_parts):
                    return False
                continue
            if i >= len(routing_parts) or routing_parts[i] != pattern_part:
                return False

        return len(routing_parts) == len(pattern_parts)


class InMemorySyncRabbitMQ(RabbitMQBase):
    """Synchronous in-memory RabbitMQ replacement"""

    def __init__(self, config: RabbitMQConfig):
        # Don't call super().__init__() as it tries to connect to real RabbitMQ
        self.config = config
        self._connection = True  # Fake connection
        self._channel = True  # Fake channel
        self._broker = InMemoryMessageBroker()

    @property
    def is_connected(self) -> bool:
        return bool(self._connection)

    @property
    def is_ready(self) -> bool:
        return bool(self._connection and self._channel)

    def connect(self) -> None:
        """Establish fake connection"""
        if self.is_connected:
            return

        logger.info("Connecting to in-memory message broker")
        self._connection = True
        self._channel = True
        self.declare_infrastructure()

    def disconnect(self) -> None:
        """Close fake connection"""
        self._connection = None
        self._channel = None

    def declare_infrastructure(self) -> None:
        """Declare exchanges and queues"""
        if not self.is_ready:
            self.connect()

        # Declare exchanges
        for exchange in self.config.exchanges:
            self._broker.declare_exchange(exchange)

        # Declare queues and bind them
        for queue in self.config.queues:
            self._broker.declare_queue(queue)
            if queue.exchange:
                self._broker.bind_queue(
                    queue.name,
                    queue.exchange.name,
                    queue.routing_key or queue.name
                )

    def publish_message(
        self,
        routing_key: str,
        message: str,
        exchange: Optional[Exchange] = None,
        properties=None,
        mandatory: bool = True,
    ) -> None:
        """Publish a message"""
        if not self.is_ready:
            self.connect()

        exchange_name = exchange.name if exchange else ""
        self._broker.publish(exchange_name, routing_key, message)

    def get_channel(self):
        """Get the channel (returns self for in-memory implementation)"""
        if not self.is_ready:
            self.connect()
        return self

    def basic_consume(
        self,
        queue: str,
        on_message_callback: Callable,
        auto_ack: bool = False
    ):
        """
        Consume messages from a queue.
        This is a simplified implementation for testing.
        """
        def consume_loop():
            while True:
                message = self._broker.get_message(queue, timeout=1.0)
                if message:
                    # Create a mock message object
                    class MockMethod:
                        delivery_tag = 1
                    class MockMessage:
                        def __init__(self, body):
                            self.body = body.encode() if isinstance(body, str) else body

                    on_message_callback(self, MockMethod(), None, MockMessage(message))

        # Start consumer in background (simplified, not production-ready)
        import threading
        thread = threading.Thread(target=consume_loop, daemon=True)
        thread.start()

    def basic_ack(self, delivery_tag):
        """Acknowledge a message (no-op for in-memory)"""
        pass

    def basic_qos(self, prefetch_count: int):
        """Set QoS (no-op for in-memory)"""
        pass


class InMemoryAsyncRabbitMQ(RabbitMQBase):
    """Asynchronous in-memory RabbitMQ replacement"""

    def __init__(self, config: RabbitMQConfig):
        # Don't call super().__init__() as it tries to connect to real RabbitMQ
        self.config = config
        self._connection = True  # Fake connection
        self._channel = True  # Fake channel
        self._broker = InMemoryMessageBroker()

    @property
    def is_connected(self) -> bool:
        return bool(self._connection)

    @property
    def is_ready(self) -> bool:
        return bool(self._connection and self._channel)

    async def connect(self):
        """Establish fake connection"""
        if self.is_connected:
            return

        logger.info("Connecting to in-memory async message broker")
        self._connection = True
        self._channel = True
        await self.declare_infrastructure()

    async def disconnect(self):
        """Close fake connection"""
        self._connection = None
        self._channel = None

    async def declare_infrastructure(self):
        """Declare exchanges and queues"""
        if not self.is_ready:
            await self.connect()

        # Declare exchanges
        for exchange in self.config.exchanges:
            self._broker.declare_exchange(exchange)

        # Declare queues and bind them
        for queue in self.config.queues:
            self._broker.declare_queue(queue)
            if queue.exchange:
                self._broker.bind_queue(
                    queue.name,
                    queue.exchange.name,
                    queue.routing_key or queue.name
                )

    async def publish_message(
        self,
        routing_key: str,
        message: str,
        exchange: Optional[Exchange] = None,
        persistent: bool = True,
    ) -> None:
        """Publish a message"""
        if not self.is_ready:
            await self.connect()

        exchange_name = exchange.name if exchange else ""
        await self._broker.publish_async(exchange_name, routing_key, message)

    async def get_channel(self):
        """Get the channel (returns self for in-memory implementation)"""
        if not self.is_ready:
            await self.connect()
        return self

    async def consume_messages(
        self,
        queue_name: str,
        callback: Callable[[str], Awaitable[None]]
    ):
        """
        Consume messages from a queue asynchronously.
        Calls the callback for each message.
        """
        while True:
            message = await self._broker.get_message_async(queue_name, timeout=1.0)
            if message:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    async def get_message(self, queue_name: str, timeout: float = 1.0) -> Optional[str]:
        """Get a single message from a queue"""
        return await self._broker.get_message_async(queue_name, timeout)
