import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from backend.data import redis
from backend.data.execution import ExecutionResult

logger = logging.getLogger(__name__)


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
