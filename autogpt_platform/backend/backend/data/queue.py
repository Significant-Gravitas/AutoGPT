import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Generator, Generic, Optional, TypeVar

from pydantic import BaseModel
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.client import PubSub

from backend.data import redis

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


M = TypeVar("M", bound=BaseModel)


class BaseRedisEventBus(Generic[M], ABC):
    Model: type[M]

    @property
    @abstractmethod
    def event_bus_name(self) -> str:
        pass

    def _serialize_message(self, item: M, channel_key: str) -> tuple[str, str]:
        message = json.dumps(item.model_dump(), cls=DateTimeEncoder)
        channel_name = f"{self.event_bus_name}/{channel_key}"
        logger.debug(f"[{channel_name}] Publishing an event to Redis {message}")
        return message, channel_name

    def _deserialize_message(self, msg: Any, channel_key: str) -> M | None:
        message_type = "pmessage" if "*" in channel_key else "message"
        if msg["type"] != message_type:
            return None
        try:
            data = json.loads(msg["data"])
            logger.debug(f"Consuming an event from Redis {data}")
            return self.Model(**data)
        except Exception as e:
            logger.error(f"Failed to parse event result from Redis {msg} {e}")

    def _get_pubsub_channel(
        self, connection: redis.Redis | redis.AsyncRedis, channel_key: str
    ) -> tuple[PubSub | AsyncPubSub, str]:
        full_channel_name = f"{self.event_bus_name}/{channel_key}"
        pubsub = connection.pubsub()
        return pubsub, full_channel_name


class RedisEventBus(BaseRedisEventBus[M], ABC):
    Model: type[M]

    @property
    def connection(self) -> redis.Redis:
        return redis.get_redis()

    def publish_event(self, event: M, channel_key: str):
        message, full_channel_name = self._serialize_message(event, channel_key)
        self.connection.publish(full_channel_name, message)

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
    Model: type[M]

    @property
    async def connection(self) -> redis.AsyncRedis:
        return await redis.get_redis_async()

    async def publish_event(self, event: M, channel_key: str):
        message, full_channel_name = self._serialize_message(event, channel_key)
        connection = await self.connection
        await connection.publish(full_channel_name, message)

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
