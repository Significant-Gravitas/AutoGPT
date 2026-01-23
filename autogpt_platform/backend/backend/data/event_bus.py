import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Generic, Optional, TypeVar

from pydantic import BaseModel
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

    def _get_pubsub_channel(
        self, connection: redis.Redis | redis.AsyncRedis, channel_key: str
    ) -> tuple[PubSub | AsyncPubSub, str]:
        full_channel_name = f"{self.event_bus_name}/{channel_key}"
        pubsub = connection.pubsub()
        return pubsub, full_channel_name


class _EventPayloadWrapper(BaseModel, Generic[M]):
    """
    Wrapper model to allow `RedisEventBus.Model` to be a discriminated union
    of multiple event types.
    """

    payload: M


class RedisEventBus(BaseRedisEventBus[M], ABC):
    @property
    def connection(self) -> redis.Redis:
        return redis.get_redis()

    def publish_event(self, event: M, channel_key: str):
        """
        Publish an event to Redis. Gracefully handles connection failures
        by logging the error instead of raising exceptions.
        """
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            self.connection.publish(full_channel_name, message)
        except Exception:
            logger.exception(
                f"Failed to publish event to Redis channel {channel_key}. "
                "Event bus operation will continue without Redis connectivity."
            )

    def listen_events(self, channel_key: str) -> Generator[M, None, None]:
        pubsub, full_channel_name = self._get_pubsub_channel(
            self.connection, channel_key
        )
        assert isinstance(pubsub, PubSub)

        if "*" in channel_key:
            pubsub.psubscribe(full_channel_name)
        else:
            pubsub.subscribe(full_channel_name)

        for message in pubsub.listen():
            if event := self._deserialize_message(message, channel_key):
                yield event


class AsyncRedisEventBus(BaseRedisEventBus[M], ABC):
    @property
    async def connection(self) -> redis.AsyncRedis:
        return await redis.get_redis_async()

    async def publish_event(self, event: M, channel_key: str):
        """
        Publish an event to Redis. Gracefully handles connection failures
        by logging the error instead of raising exceptions.
        """
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            connection = await self.connection
            await connection.publish(full_channel_name, message)
        except Exception:
            logger.exception(
                f"Failed to publish event to Redis channel {channel_key}. "
                "Event bus operation will continue without Redis connectivity."
            )

    async def listen_events(self, channel_key: str) -> AsyncGenerator[M, None]:
        pubsub, full_channel_name = self._get_pubsub_channel(
            await self.connection, channel_key
        )
        assert isinstance(pubsub, AsyncPubSub)

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
