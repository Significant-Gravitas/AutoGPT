import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Generator, Generic, TypeVar

from pydantic import BaseModel
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.client import PubSub

from backend.data import redis
from backend.data.execution import ExecutionResult
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()


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
        channel_name = f"{self.event_bus_name}-{channel_key}"
        logger.info(f"[{channel_name}] Publishing an event to Redis {message}")
        return message, channel_name

    def _deserialize_message(self, msg: Any, channel_key: str) -> M | None:
        message_type = "pmessage" if "*" in channel_key else "message"
        if msg["type"] != message_type:
            return None
        try:
            data = json.loads(msg["data"])
            logger.info(f"Consuming an event from Redis {data}")
            return self.Model(**data)
        except Exception as e:
            logger.error(f"Failed to parse event result from Redis {msg} {e}")

    def _subscribe(
        self, connection: redis.Redis | redis.AsyncRedis, channel_key: str
    ) -> tuple[PubSub | AsyncPubSub, str]:
        channel_name = f"{self.event_bus_name}-{channel_key}"
        pubsub = connection.pubsub()
        return pubsub, channel_name


class RedisEventBus(BaseRedisEventBus[M], ABC):
    Model: type[M]

    @property
    def connection(self) -> redis.Redis:
        return redis.get_redis()

    def publish_event(self, event: M, channel_key: str):
        message, channel_name = self._serialize_message(event, channel_key)
        self.connection.publish(channel_name, message)

    def listen_events(self, channel_key: str) -> Generator[M, None, None]:
        pubsub, channel_name = self._subscribe(self.connection, channel_key)
        assert isinstance(pubsub, PubSub)

        if "*" in channel_key:
            pubsub.psubscribe(channel_name)
        else:
            pubsub.subscribe(channel_name)

        for message in pubsub.listen():
            if event := self._deserialize_message(message, channel_key):
                yield event


class AsyncRedisEventBus(BaseRedisEventBus[M], ABC):
    Model: type[M]

    @property
    async def connection(self) -> redis.AsyncRedis:
        return await redis.get_redis_async()

    async def publish_event(self, event: M, channel_key: str):
        message, channel_name = self._serialize_message(event, channel_key)
        connection = await self.connection
        await connection.publish(channel_name, message)

    async def listen_events(self, channel_key: str) -> AsyncGenerator[M, None]:
        pubsub, channel_name = self._subscribe(await self.connection, channel_key)
        assert isinstance(pubsub, AsyncPubSub)

        if "*" in channel_key:
            await pubsub.psubscribe(channel_name)
        else:
            await pubsub.subscribe(channel_name)

        async for message in pubsub.listen():
            if event := self._deserialize_message(message, channel_key):
                yield event


class RedisExecutionEventBus(RedisEventBus[ExecutionResult]):
    Model = ExecutionResult

    @property
    def event_bus_name(self) -> str:
        return config.execution_event_bus_name

    def publish(self, res: ExecutionResult):
        self.publish_event(res, f"{res.graph_id}-{res.graph_exec_id}")

    def listen(
        self, graph_id: str = "*", graph_exec_id: str = "*"
    ) -> Generator[ExecutionResult, None, None]:
        for execution_result in self.listen_events(f"{graph_id}-{graph_exec_id}"):
            yield execution_result


class AsyncRedisExecutionEventBus(AsyncRedisEventBus[ExecutionResult]):
    Model = ExecutionResult

    @property
    def event_bus_name(self) -> str:
        return config.execution_event_bus_name

    async def publish(self, res: ExecutionResult):
        await self.publish_event(res, f"{res.graph_id}-{res.graph_exec_id}")

    async def listen(
        self, graph_id: str = "*", graph_exec_id: str = "*"
    ) -> AsyncGenerator[ExecutionResult, None]:
        async for execution_result in self.listen_events(f"{graph_id}-{graph_exec_id}"):
            yield execution_result
