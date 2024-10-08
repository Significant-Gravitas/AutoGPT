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
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def put(self, execution_result: ExecutionResult):
        pass

    @abstractmethod
    def get(self) -> ExecutionResult | None:
        pass


class RedisEventQueue(AbstractEventQueue):
    def __init__(self):
        self.connection = None
        self.queue_name = redis.QUEUE_NAME

    def connect(self):
        self.connection = redis.connect()

    def put(self, execution_result: ExecutionResult):
        if self.connection:
            message = json.dumps(execution_result.model_dump(), cls=DateTimeEncoder)
            logger.info(f"Putting execution result to Redis {message}")
            self.connection.lpush(self.queue_name, message)

    def get(self) -> ExecutionResult | None:
        if self.connection:
            message = self.connection.rpop(self.queue_name)
            if message is not None and isinstance(message, (str, bytes, bytearray)):
                data = json.loads(message)
                logger.info(f"Getting execution result from Redis {data}")
                return ExecutionResult(**data)
        return None

    def close(self):
        redis.disconnect()
