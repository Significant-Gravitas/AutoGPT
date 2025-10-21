"""
In-memory event bus implementation as a fallback for Redis.
This allows running the platform without an external Redis server.
"""
import asyncio
import logging
from abc import ABC
from collections import defaultdict
from typing import Any, AsyncGenerator, Generator, Generic
from queue import Queue, Empty
from threading import Lock

from pydantic import BaseModel

from backend.util import json
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
config = Settings().config

# Type variable for message model
M = type(BaseModel)


class _EventPayloadWrapper(BaseModel, Generic[M]):
    """Wrapper model to allow message payloads"""
    payload: M


class InMemoryEventBus:
    """
    Singleton in-memory event bus that stores all subscribers and messages.
    Thread-safe implementation using locks and queues.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._subscribers = defaultdict(list)
                    cls._instance._async_subscribers = defaultdict(list)
                    cls._instance._subscriber_lock = Lock()
        return cls._instance

    def publish(self, channel: str, message: str):
        """Publish a message to all subscribers of a channel"""
        with self._subscriber_lock:
            # Handle pattern subscriptions (channels with *)
            for pattern, queues in self._subscribers.items():
                if self._match_pattern(pattern, channel):
                    for queue in queues:
                        try:
                            queue.put({
                                'type': 'pmessage' if '*' in pattern else 'message',
                                'channel': channel,
                                'data': message
                            })
                        except Exception as e:
                            logger.error(f"Failed to publish to queue: {e}")

    async def publish_async(self, channel: str, message: str):
        """Async version of publish"""
        with self._subscriber_lock:
            # Notify async subscribers
            for pattern, queues in self._async_subscribers.items():
                if self._match_pattern(pattern, channel):
                    for queue in queues:
                        try:
                            await queue.put({
                                'type': 'pmessage' if '*' in pattern else 'message',
                                'channel': channel,
                                'data': message
                            })
                        except Exception as e:
                            logger.error(f"Failed to publish async to queue: {e}")

    def subscribe(self, channel: str) -> Queue:
        """Subscribe to a channel and return a queue for receiving messages"""
        queue = Queue()
        with self._subscriber_lock:
            self._subscribers[channel].append(queue)
        return queue

    async def subscribe_async(self, channel: str) -> asyncio.Queue:
        """Async version of subscribe"""
        queue = asyncio.Queue()
        with self._subscriber_lock:
            self._async_subscribers[channel].append(queue)
        return queue

    def unsubscribe(self, channel: str, queue: Queue):
        """Unsubscribe a queue from a channel"""
        with self._subscriber_lock:
            if channel in self._subscribers:
                try:
                    self._subscribers[channel].remove(queue)
                except ValueError:
                    pass

    async def unsubscribe_async(self, channel: str, queue: asyncio.Queue):
        """Async version of unsubscribe"""
        with self._subscriber_lock:
            if channel in self._async_subscribers:
                try:
                    self._async_subscribers[channel].remove(queue)
                except ValueError:
                    pass

    @staticmethod
    def _match_pattern(pattern: str, channel: str) -> bool:
        """Match a channel against a pattern (supports * wildcard)"""
        if '*' not in pattern:
            return pattern == channel

        # Simple pattern matching for Redis-style patterns
        pattern_parts = pattern.split('*')
        if len(pattern_parts) == 2:
            prefix, suffix = pattern_parts
            return channel.startswith(prefix) and channel.endswith(suffix)

        # More complex patterns - just check prefix for now
        return channel.startswith(pattern_parts[0])


class BaseInMemoryEventBus(Generic[M], ABC):
    """Base class for in-memory event bus implementations"""
    Model: type[M]

    @property
    def event_bus_name(self) -> str:
        """Override this in subclasses"""
        return "events"

    @property
    def Message(self) -> type[_EventPayloadWrapper[M]]:
        return _EventPayloadWrapper[self.Model]

    def _serialize_message(self, item: M, channel_key: str) -> tuple[str, str]:
        """Serialize a message for publishing"""
        MAX_MESSAGE_SIZE = config.max_message_size_limit

        try:
            message = json.dumps(
                self.Message(payload=item), ensure_ascii=False, separators=(",", ":")
            )
        except UnicodeError:
            message = json.dumps(
                self.Message(payload=item), ensure_ascii=True, separators=(",", ":")
            )
            logger.warning(
                f"Unicode serialization failed, falling back to ASCII for channel {channel_key}"
            )

        # Check message size and truncate if necessary
        message_size = len(message.encode("utf-8"))
        if message_size > MAX_MESSAGE_SIZE:
            logger.warning(
                f"Message size {message_size} bytes exceeds limit {MAX_MESSAGE_SIZE} bytes for channel {channel_key}. "
                "Truncating payload."
            )
            error_payload = {
                "payload": {
                    "event_type": "error_comms_update",
                    "error": "Payload too large for transmission",
                    "original_size_bytes": message_size,
                    "max_size_bytes": MAX_MESSAGE_SIZE,
                }
            }
            message = json.dumps(
                error_payload, ensure_ascii=False, separators=(",", ":")
            )

        channel_name = f"{self.event_bus_name}/{channel_key}"
        logger.debug(f"[{channel_name}] Publishing event: {message[:100]}...")
        return message, channel_name

    def _deserialize_message(self, msg: dict, channel_key: str) -> M | None:
        """Deserialize a message from the queue"""
        message_type = "pmessage" if "*" in channel_key else "message"
        if msg["type"] != message_type:
            return None
        try:
            logger.debug(f"[{channel_key}] Consuming event: {msg['data'][:100]}...")
            return self.Message.model_validate_json(msg["data"]).payload
        except Exception as e:
            logger.error(f"Failed to parse event from queue {msg}: {e}")
            return None


class InMemorySyncEventBus(BaseInMemoryEventBus[M], ABC):
    """Synchronous in-memory event bus"""

    def __init__(self):
        self._bus = InMemoryEventBus()

    def publish_event(self, event: M, channel_key: str):
        """Publish an event to a channel"""
        message, full_channel_name = self._serialize_message(event, channel_key)
        self._bus.publish(full_channel_name, message)

    def listen_events(self, channel_key: str) -> Generator[M, None, None]:
        """Listen for events on a channel"""
        full_channel_name = f"{self.event_bus_name}/{channel_key}"
        queue = self._bus.subscribe(full_channel_name)

        try:
            while True:
                try:
                    message = queue.get(timeout=1.0)
                    if event := self._deserialize_message(message, channel_key):
                        yield event
                except Empty:
                    continue
        finally:
            self._bus.unsubscribe(full_channel_name, queue)


class InMemoryAsyncEventBus(BaseInMemoryEventBus[M], ABC):
    """Asynchronous in-memory event bus"""

    def __init__(self):
        self._bus = InMemoryEventBus()

    async def publish_event(self, event: M, channel_key: str):
        """Publish an event to a channel"""
        message, full_channel_name = self._serialize_message(event, channel_key)
        await self._bus.publish_async(full_channel_name, message)

    async def listen_events(self, channel_key: str) -> AsyncGenerator[M, None]:
        """Listen for events on a channel"""
        full_channel_name = f"{self.event_bus_name}/{channel_key}"
        queue = await self._bus.subscribe_async(full_channel_name)

        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if event := self._deserialize_message(message, channel_key):
                        yield event
                except asyncio.TimeoutError:
                    continue
        finally:
            await self._bus.unsubscribe_async(full_channel_name, queue)

    async def wait_for_event(
        self, channel_key: str, timeout: float | None = None
    ) -> M | None:
        """Wait for a single event with optional timeout"""
        try:
            return await asyncio.wait_for(
                anext(aiter(self.listen_events(channel_key))), timeout
            )
        except (TimeoutError, asyncio.TimeoutError):
            return None
