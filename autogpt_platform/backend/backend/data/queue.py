import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from backend.data import redis
from backend.data.execution import ExecutionResult

logger = logging.getLogger(__name__)

T = TypeVar("T")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class AbstractEventQueue(ABC):
    @abstractmethod
    def put(self, execution_result: ExecutionResult):
        pass

    @abstractmethod
    def get(self) -> ExecutionResult | None:
        pass


class RedisEventQueue(AbstractEventQueue):
    def __init__(self):
        self.queue_name = redis.QUEUE_NAME

    @property
    def connection(self):
        return redis.get_redis()

    def put(self, execution_result: ExecutionResult):
        message = json.dumps(execution_result.model_dump(), cls=DateTimeEncoder)
        logger.info(f"Putting execution result to Redis {message}")
        self.connection.lpush(self.queue_name, message)

    def get(self) -> ExecutionResult | None:
        message = self.connection.rpop(self.queue_name)
        if message is not None and isinstance(message, (str, bytes, bytearray)):
            data = json.loads(message)
            logger.info(f"Getting execution result from Redis {data}")
            return ExecutionResult(**data)
        elif message is not None:
            logger.error(f"Failed to get execution result from Redis {message}")
        return None


class ExecutionQueue(Generic[T]):
    def __init__(self, queue_name: str):
        self.redis = redis.get_redis()
        self.queue_name = queue_name

    def add(self, item: T):
        message = json.dumps(item.model_dump(), default=str)
        self.redis.lpush(self.queue_name, message)

    def get(self) -> T:
        while True:
            _, message = self.redis.brpop(self.queue_name)
            return T.model_validate(json.loads(message))

    def empty(self) -> bool:
        return self.redis.llen(self.queue_name) == 0
