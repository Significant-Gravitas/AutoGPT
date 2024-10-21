import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Generator

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


class BaseRedisEventBus:

    def __init__(self):
        self.event_bus_name = config.execution_event_bus_name

    def _serialize_message(self, execution_result: ExecutionResult) -> tuple[str, str]:
        message = json.dumps(execution_result.model_dump(), cls=DateTimeEncoder)
        logger.info(f"Putting execution result to Redis {message}")
        channel_name = f"{self.event_bus_name}-{execution_result.graph_id}-{execution_result.graph_exec_id}"
        return message, channel_name

    @staticmethod
    def _deserialize_message(msg: Any) -> ExecutionResult | None:
        if msg["type"] not in ("message", "pmessage"):
            return None
        try:
            data = json.loads(msg["data"])
            logger.info(f"Getting execution result from Redis {data}")
            return ExecutionResult(**data)
        except Exception as e:
            logger.error(f"Failed to get execution result from Redis {msg} {e}")


class RedisEventBus(BaseRedisEventBus):
    @property
    def connection(self) -> redis.Redis:
        return redis.get_redis()

    def publish(self, execution_result: ExecutionResult):
        message, channel_name = self._serialize_message(execution_result)
        self.connection.publish(channel_name, message)

    def listen(
        self, graph_id: str = "*", execution_id: str = "*"
    ) -> Generator[ExecutionResult, None, None]:
        pubsub = self.connection.pubsub()
        channel_name = f"{self.event_bus_name}-{graph_id}-{execution_id}"
        if "*" in channel_name:
            pubsub.psubscribe(channel_name)
        else:
            pubsub.subscribe(channel_name)

        for message in pubsub.listen():
            if execution_result := self._deserialize_message(message):
                yield execution_result


class AsyncRedisEventBus(BaseRedisEventBus):
    @property
    async def connection(self) -> redis.AsyncRedis:
        return await redis.get_redis_async()

    async def publish(self, execution_result: ExecutionResult):
        message, channel_name = self._serialize_message(execution_result)
        connection = await self.connection
        await connection.publish(channel_name, message)

    async def listen(
        self, graph_id: str = "*", execution_id: str = "*"
    ) -> AsyncGenerator[ExecutionResult, None]:
        connection = await self.connection
        pubsub = connection.pubsub()
        channel_name = f"{self.event_bus_name}-{graph_id}-{execution_id}"
        if "*" in channel_name:
            await pubsub.psubscribe(channel_name)
        else:
            await pubsub.subscribe(channel_name)

        async for message in pubsub.listen():
            if execution_result := self._deserialize_message(message):
                yield execution_result
