import json
import logging
import os
from abc import abstractmethod, ABC
from typing import Any
import aio_pika
from autogpt_server.data.execution import ExecutionResult


class AsyncEventQueue(ABC):
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def close(self):
        pass

    @abstractmethod
    async def get_execution_result(self) -> ExecutionResult:
        pass

    @abstractmethod
    async def publish_execution_result(self, result: ExecutionResult):
        pass

class AsyncRabbitMQEventQueue:
    def __init__(self):
        self.host = os.getenv('RABBITMQ_HOST', 'localhost')
        self.port = int(os.getenv('RABBITMQ_PORT', '5672'))
        self.user = os.getenv('RABBITMQ_USER', 'guest')
        self.password = os.getenv('RABBITMQ_PASSWORD', 'guest')
        self.vhost = os.getenv('RABBITMQ_VHOST', '/')
        self.queue_name = os.getenv('RABBITMQ_QUEUE', 'execution_events')
        self.connection = None
        self.channel = None
        self.queue = None

    async def connect(self):
        try:
            self.connection = await aio_pika.connect_robust(
                f"amqp://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}"
            )
            self.channel = await self.connection.channel()
            self.queue = await self.channel.declare_queue(self.queue_name)
            logging.info(f"Connected to RabbitMQ and declared queue: {self.queue_name}")
        except Exception as e:
            logging.exception(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    async def close(self):
        if self.connection:
            await self.connection.close()
            logging.info("Closed connection to RabbitMQ")

    async def publish_execution_result(self, result: ExecutionResult):
        if not self.channel:
            await self.connect()

        message = aio_pika.Message(body=json.dumps({
            'graph_id': result.graph_id,
            'result': result.model_dump()
        }).encode())

        await self.channel.default_exchange.publish(
            message, routing_key=self.queue_name
        )

    async def get_execution_result(self) -> ExecutionResult:
        if not self.queue:
            await self.connect()

        async with self.queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body.decode())
                    return ExecutionResult(**data['result'])