import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

from redis.asyncio import Redis

from autogpt_server.data.execution import ExecutionResult

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class AsyncEventQueue(ABC):
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    async def put(self, execution_result: ExecutionResult):
        pass

    @abstractmethod
    async def get(self) -> ExecutionResult | None:
        pass


class AsyncRedisEventQueue(AsyncEventQueue):

    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.password = os.getenv("REDIS_PASSWORD", "password")
        self.queue_name = os.getenv("REDIS_QUEUE", "execution_events")
        self.connection = None

    async def connect(self):
        if not self.connection:
            self.connection = Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
            )
            await self.connection.ping()
            logger.info(f"Connected to Redis on {self.host}:{self.port}")

    async def put(self, execution_result: ExecutionResult):
        if self.connection:
            message = json.dumps(execution_result.model_dump(), cls=DateTimeEncoder)
            logger.info(f"Putting execution result to Redis {message}")
            await self.connection.lpush(self.queue_name, message)  # type: ignore

    async def get(self) -> ExecutionResult | None:
        if self.connection:
            message = await self.connection.rpop(self.queue_name)  # type: ignore
            if message is not None and isinstance(message, (str, bytes, bytearray)):
                data = json.loads(message)
                logger.info(f"Getting execution result from Redis {data}")
                return ExecutionResult(**data)
        return None

    async def close(self):
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Closed connection to Redis")
