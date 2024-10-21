import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

from backend.data import redis
from backend.data.execution import ExecutionResult

M = TypeVar("M", bound=BaseModel)

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class AbstractEventQueue(ABC, Generic[M]):
    @abstractmethod
    def put(self, item: M):
        pass

    @abstractmethod
    def get(self) -> M | None:
        pass


class RedisEventQueue(AbstractEventQueue[M]):
    Model: type[M]

    def __init__(self):
        self.queue_name = redis.QUEUE_NAME

    @property
    def connection(self):
        return redis.get_redis()

    def put(self, item: M):
        message = json.dumps(item.model_dump(), cls=DateTimeEncoder)
        logger.info(f"Putting item to Redis queue [{self.queue_name}]: {message}")
        self.connection.lpush(self.queue_name, message)

    def get(self) -> M | None:
        message = self.connection.rpop(self.queue_name)
        if message is not None and isinstance(message, (str, bytes, bytearray)):
            data = json.loads(message)
            logger.info(f"Getting item from Redis queue [{self.queue_name}]: {data}")
            return self.Model(**data)
        elif message is not None:
            logger.error(
                f"Failed to get item from Redis queue [{self.queue_name}]: {message}"
            )
        return None


class RedisExecutionEventQueue(RedisEventQueue[ExecutionResult]):
    Model = ExecutionResult
