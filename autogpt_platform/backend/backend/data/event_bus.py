import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Generic, Optional, TypeVar

from pydantic import BaseModel
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.client import PubSub

from backend.data import redis_client as redis
from backend.util import json
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
config = Settings().config


M = TypeVar("M", bound=BaseModel)


class BaseRedisEventBus(Generic[M], ABC):
    Model: type[M]

    @property
    @abstractmethod
    def event_bus_name(self) -> str:
        pass

    @property
    def Message(self) -> type["_EventPayloadWrapper[M]"]:
        return _EventPayloadWrapper[self.Model]

    def _serialize_message(self, item: M, channel_key: str) -> tuple[str, str]:
        MAX_MESSAGE_SIZE = config.max_message_size_limit

        try:
            # Use backend.util.json.dumps which handles datetime and other complex types
            message = json.dumps(
                self.Message(payload=item), ensure_ascii=False, separators=(",", ":")
            )
        except UnicodeError:
            # Fallback to ASCII encoding if Unicode causes issues
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
                "Truncating payload to prevent Redis connection issues."
            )
            error_payload = {
                "payload": {
                    "event_type": "error_comms_update",
                    "error": "Payload too large for Redis transmission",
                    "original_size_bytes": message_size,
                    "max_size_bytes": MAX_MESSAGE_SIZE,
                }
            }
            message = json.dumps(
                error_payload, ensure_ascii=False, separators=(",", ":")
            )

        channel_name = f"{self.event_bus_name}/{channel_key}"
        logger.debug(f"[{channel_name}] Publishing an event to Redis {message}")
        return message, channel_name

    def _deserialize_message(self, msg: Any, channel_key: str) -> M | None:
        message_type = "pmessage" if "*" in channel_key else "message"
        if msg["type"] != message_type:
            return None
        try:
            logger.debug(f"[{channel_key}] Consuming an event from Redis {msg['data']}")
            return self.Message.model_validate_json(msg["data"]).payload
        except Exception as e:
            logger.error(f"Failed to parse event result from Redis {msg} {e}")

    def _build_channel_name(self, channel_key: str) -> str:
        return f"{self.event_bus_name}/{channel_key}"


class _EventPayloadWrapper(BaseModel, Generic[M]):
    """
    Wrapper model to allow `RedisEventBus.Model` to be a discriminated union
    of multiple event types.
    """

    payload: M


class RedisEventBus(BaseRedisEventBus[M], ABC):
    @property
    def pubsub_connection(self) -> Redis:
        # Dedicated standalone client: a subscribed connection blocks on
        # ``listen()`` and cannot share a socket with regular commands.
        return redis.get_redis_pubsub()

    def publish_event(self, event: M, channel_key: str):
        """
        Publish an event to Redis. Gracefully handles connection failures
        by logging the error instead of raising exceptions.
        """
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            self.pubsub_connection.publish(full_channel_name, message)
        except Exception:
            logger.exception(
                f"Failed to publish event to Redis channel {channel_key}. "
                "Event bus operation will continue without Redis connectivity."
            )

    def listen_events(self, channel_key: str) -> Generator[M, None, None]:
        full_channel_name = self._build_channel_name(channel_key)
        pubsub: PubSub = self.pubsub_connection.pubsub()

        if "*" in channel_key:
            pubsub.psubscribe(full_channel_name)
        else:
            pubsub.subscribe(full_channel_name)

        for message in pubsub.listen():
            if event := self._deserialize_message(message, channel_key):
                yield event


class AsyncRedisEventBus(BaseRedisEventBus[M], ABC):
    def __init__(self):
        self._pubsub: AsyncPubSub | None = None

    async def get_pubsub_connection(self) -> AsyncRedis:
        # Plain method (not ``@property``) to keep the async-ness visible at
        # the call site. Standalone client is mandatory because async
        # RedisCluster has no ``pubsub()``.
        return await redis.get_redis_pubsub_async()

    async def close(self) -> None:
        """Close the PubSub connection if it exists."""
        if self._pubsub is not None:
            try:
                await self._pubsub.close()
            except Exception:
                logger.warning("Failed to close PubSub connection", exc_info=True)
            finally:
                self._pubsub = None

    async def publish_event(self, event: M, channel_key: str):
        """
        Publish an event to Redis. Gracefully handles connection failures
        by logging the error instead of raising exceptions.
        """
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            connection = await self.get_pubsub_connection()
            await connection.publish(full_channel_name, message)
        except Exception:
            logger.exception(
                f"Failed to publish event to Redis channel {channel_key}. "
                "Event bus operation will continue without Redis connectivity."
            )

    async def listen_events(self, channel_key: str) -> AsyncGenerator[M, None]:
        connection = await self.get_pubsub_connection()
        full_channel_name = self._build_channel_name(channel_key)
        pubsub: AsyncPubSub = connection.pubsub()
        self._pubsub = pubsub

        if "*" in channel_key:
            await pubsub.psubscribe(full_channel_name)
        else:
            await pubsub.subscribe(full_channel_name)

        async for message in pubsub.listen():
            if event := self._deserialize_message(message, channel_key):
                yield event

    async def wait_for_event(
        self, channel_key: str, timeout: Optional[float] = None
    ) -> M | None:
        try:
            return await asyncio.wait_for(
                anext(aiter(self.listen_events(channel_key))), timeout
            )
        except TimeoutError:
            return None
