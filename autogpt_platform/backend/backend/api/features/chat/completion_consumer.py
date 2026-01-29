"""RabbitMQ consumer for operation completion messages.

This module provides a consumer that listens for completion notifications
from external services (like Agent Generator) and triggers the appropriate
stream registry and chat service updates.
"""

import asyncio
import logging
from typing import Any

import orjson
from pydantic import BaseModel

from backend.data.rabbitmq import AsyncRabbitMQ, Exchange, ExchangeType, Queue, RabbitMQConfig

from . import service as chat_service
from . import stream_registry
from .response_model import StreamError, StreamToolOutputAvailable
from .tools.models import ErrorResponse

logger = logging.getLogger(__name__)

# Queue and exchange configuration
OPERATION_COMPLETE_EXCHANGE = Exchange(
    name="chat_operations",
    type=ExchangeType.DIRECT,
    durable=True,
)

OPERATION_COMPLETE_QUEUE = Queue(
    name="chat_operation_complete",
    durable=True,
    exchange=OPERATION_COMPLETE_EXCHANGE,
    routing_key="operation.complete",
)

RABBITMQ_CONFIG = RabbitMQConfig(
    exchanges=[OPERATION_COMPLETE_EXCHANGE],
    queues=[OPERATION_COMPLETE_QUEUE],
)


class OperationCompleteMessage(BaseModel):
    """Message format for operation completion notifications."""

    operation_id: str
    task_id: str
    success: bool
    result: dict | str | None = None
    error: str | None = None


class ChatCompletionConsumer:
    """Consumer for chat operation completion messages from RabbitMQ."""

    def __init__(self):
        self._rabbitmq: AsyncRabbitMQ | None = None
        self._consumer_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the completion consumer."""
        if self._running:
            logger.warning("Completion consumer already running")
            return

        self._rabbitmq = AsyncRabbitMQ(RABBITMQ_CONFIG)
        await self._rabbitmq.connect()

        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_messages())
        logger.info("Chat completion consumer started")

    async def stop(self) -> None:
        """Stop the completion consumer."""
        self._running = False

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        if self._rabbitmq:
            await self._rabbitmq.disconnect()
            self._rabbitmq = None

        logger.info("Chat completion consumer stopped")

    async def _consume_messages(self) -> None:
        """Main message consumption loop."""
        if not self._rabbitmq:
            logger.error("RabbitMQ not initialized")
            return

        try:
            channel = await self._rabbitmq.get_channel()
            queue = await channel.get_queue(OPERATION_COMPLETE_QUEUE.name)

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    if not self._running:
                        break

                    try:
                        async with message.process():
                            await self._handle_message(message.body)
                    except Exception as e:
                        logger.error(
                            f"Error processing completion message: {e}",
                            exc_info=True,
                        )
                        # Message will be requeued due to exception

        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            # Attempt to reconnect after a delay
            if self._running:
                await asyncio.sleep(5)
                await self._consume_messages()

    async def _handle_message(self, body: bytes) -> None:
        """Handle a single completion message."""
        try:
            data = orjson.loads(body)
            message = OperationCompleteMessage(**data)
        except Exception as e:
            logger.error(f"Failed to parse completion message: {e}")
            return

        logger.info(
            f"Received completion for operation {message.operation_id} "
            f"(task_id={message.task_id}, success={message.success})"
        )

        # Find task in registry
        task = await stream_registry.find_task_by_operation_id(message.operation_id)
        if task is None:
            # Try to look up by task_id directly
            task = await stream_registry.get_task(message.task_id)

        if task is None:
            logger.warning(
                f"Task not found for operation {message.operation_id} "
                f"(task_id={message.task_id})"
            )
            return

        if message.success:
            await self._handle_success(task, message)
        else:
            await self._handle_failure(task, message)

    async def _handle_success(
        self,
        task: stream_registry.ActiveTask,
        message: OperationCompleteMessage,
    ) -> None:
        """Handle successful operation completion."""
        # Publish result to stream registry
        result_output = message.result if message.result else {"status": "completed"}
        await stream_registry.publish_chunk(
            task.task_id,
            StreamToolOutputAvailable(
                toolCallId=task.tool_call_id,
                toolName=task.tool_name,
                output=(
                    result_output
                    if isinstance(result_output, str)
                    else orjson.dumps(result_output).decode("utf-8")
                ),
                success=True,
            ),
        )

        # Update pending operation in database
        result_str = (
            message.result
            if isinstance(message.result, str)
            else orjson.dumps(message.result).decode("utf-8")
            if message.result
            else '{"status": "completed"}'
        )
        await chat_service._update_pending_operation(
            session_id=task.session_id,
            tool_call_id=task.tool_call_id,
            result=result_str,
        )

        # Generate LLM continuation with streaming
        await chat_service._generate_llm_continuation_with_streaming(
            session_id=task.session_id,
            user_id=task.user_id,
            task_id=task.task_id,
        )

        # Mark task as completed
        await stream_registry.mark_task_completed(task.task_id, status="completed")

        logger.info(
            f"Successfully processed completion for task {task.task_id} "
            f"(operation {message.operation_id})"
        )

    async def _handle_failure(
        self,
        task: stream_registry.ActiveTask,
        message: OperationCompleteMessage,
    ) -> None:
        """Handle failed operation completion."""
        error_msg = message.error or "Operation failed"

        # Publish error to stream registry
        await stream_registry.publish_chunk(
            task.task_id,
            StreamError(errorText=error_msg),
        )

        # Update pending operation with error
        error_response = ErrorResponse(
            message=error_msg,
            error=message.error,
        )
        await chat_service._update_pending_operation(
            session_id=task.session_id,
            tool_call_id=task.tool_call_id,
            result=error_response.model_dump_json(),
        )

        # Mark task as failed
        await stream_registry.mark_task_completed(task.task_id, status="failed")

        logger.info(
            f"Processed failure for task {task.task_id} "
            f"(operation {message.operation_id}): {error_msg}"
        )


# Module-level consumer instance
_consumer: ChatCompletionConsumer | None = None


async def start_completion_consumer() -> None:
    """Start the global completion consumer."""
    global _consumer
    if _consumer is None:
        _consumer = ChatCompletionConsumer()
    await _consumer.start()


async def stop_completion_consumer() -> None:
    """Stop the global completion consumer."""
    global _consumer
    if _consumer:
        await _consumer.stop()
        _consumer = None


async def publish_operation_complete(
    operation_id: str,
    task_id: str,
    success: bool,
    result: dict | str | None = None,
    error: str | None = None,
) -> None:
    """Publish an operation completion message.

    This is a helper function for testing or for services that want to
    publish completion messages directly.

    Args:
        operation_id: The operation ID that completed.
        task_id: The task ID associated with the operation.
        success: Whether the operation succeeded.
        result: The result data (for success).
        error: The error message (for failure).
    """
    message = OperationCompleteMessage(
        operation_id=operation_id,
        task_id=task_id,
        success=success,
        result=result,
        error=error,
    )

    rabbitmq = AsyncRabbitMQ(RABBITMQ_CONFIG)
    try:
        await rabbitmq.connect()
        await rabbitmq.publish_message(
            routing_key="operation.complete",
            message=message.model_dump_json(),
            exchange=OPERATION_COMPLETE_EXCHANGE,
        )
        logger.info(f"Published completion for operation {operation_id}")
    finally:
        await rabbitmq.disconnect()
