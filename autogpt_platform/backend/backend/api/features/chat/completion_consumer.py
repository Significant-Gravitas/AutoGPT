"""Redis Streams consumer for operation completion messages.

This module provides a consumer that listens for completion notifications
from external services (like Agent Generator) and triggers the appropriate
stream registry and chat service updates.

The consumer uses Redis Streams with consumer groups for reliable message
processing across multiple platform pods.
"""

import asyncio
import logging
import os
import uuid

import orjson
from prisma import Prisma
from pydantic import BaseModel
from redis.exceptions import ResponseError

from backend.data.redis_client import get_redis_async

from . import service as chat_service
from . import stream_registry
from .response_model import StreamError, StreamFinish, StreamToolOutputAvailable
from .tools.models import ErrorResponse

logger = logging.getLogger(__name__)

# Stream configuration
COMPLETION_STREAM = "chat:completions"
CONSUMER_GROUP = "chat_consumers"
STREAM_MAX_LENGTH = 10000


class OperationCompleteMessage(BaseModel):
    """Message format for operation completion notifications."""

    operation_id: str
    task_id: str
    success: bool
    result: dict | str | None = None
    error: str | None = None


class ChatCompletionConsumer:
    """Consumer for chat operation completion messages from Redis Streams.

    This consumer initializes its own Prisma client in start() to ensure
    database operations work correctly within this async context.

    Uses Redis consumer groups to allow multiple platform pods to consume
    messages reliably with automatic redelivery on failure.
    """

    def __init__(self):
        self._consumer_task: asyncio.Task | None = None
        self._running = False
        self._prisma: Prisma | None = None
        self._consumer_name = f"consumer-{uuid.uuid4().hex[:8]}"

    async def start(self) -> None:
        """Start the completion consumer."""
        if self._running:
            logger.warning("Completion consumer already running")
            return

        # Create consumer group if it doesn't exist
        try:
            redis = await get_redis_async()
            await redis.xgroup_create(
                COMPLETION_STREAM,
                CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info(
                f"Created consumer group '{CONSUMER_GROUP}' on stream '{COMPLETION_STREAM}'"
            )
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group '{CONSUMER_GROUP}' already exists")
            else:
                raise

        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_messages())
        logger.info(
            f"Chat completion consumer started (consumer: {self._consumer_name})"
        )

    async def _ensure_prisma(self) -> Prisma:
        """Lazily initialize Prisma client on first use."""
        if self._prisma is None:
            database_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432")
            self._prisma = Prisma(datasource={"url": database_url})
            await self._prisma.connect()
            logger.info("[COMPLETION] Consumer Prisma client connected (lazy init)")
        return self._prisma

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

        if self._prisma:
            await self._prisma.disconnect()
            self._prisma = None
            logger.info("[COMPLETION] Consumer Prisma client disconnected")

        logger.info("Chat completion consumer stopped")

    async def _consume_messages(self) -> None:
        """Main message consumption loop with retry logic."""
        max_retries = 10
        retry_delay = 5  # seconds
        retry_count = 0
        block_timeout = 5000  # milliseconds

        while self._running and retry_count < max_retries:
            try:
                redis = await get_redis_async()

                # Reset retry count on successful connection
                retry_count = 0

                while self._running:
                    # Read new messages from the stream
                    messages = await redis.xreadgroup(
                        groupname=CONSUMER_GROUP,
                        consumername=self._consumer_name,
                        streams={COMPLETION_STREAM: ">"},
                        block=block_timeout,
                        count=10,
                    )

                    if not messages:
                        continue

                    for stream_name, entries in messages:
                        for entry_id, data in entries:
                            if not self._running:
                                return

                            try:
                                # Handle the message
                                message_data = data.get("data")
                                if message_data:
                                    await self._handle_message(
                                        message_data.encode()
                                        if isinstance(message_data, str)
                                        else message_data
                                    )

                                # Acknowledge the message
                                await redis.xack(
                                    COMPLETION_STREAM, CONSUMER_GROUP, entry_id
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing completion message {entry_id}: {e}",
                                    exc_info=True,
                                )
                                # Message will be redelivered to another consumer
                                # or can be claimed after timeout

            except asyncio.CancelledError:
                logger.info("Consumer cancelled")
                return
            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Consumer error (retry {retry_count}/{max_retries}): {e}",
                    exc_info=True,
                )
                if self._running and retry_count < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached, stopping consumer")
                    return

    async def _handle_message(self, body: bytes) -> None:
        """Handle a completion message using our own Prisma client."""
        try:
            data = orjson.loads(body)
            message = OperationCompleteMessage(**data)
        except Exception as e:
            logger.error(f"Failed to parse completion message: {e}")
            return

        logger.info(
            f"[COMPLETION] Received completion for operation {message.operation_id} "
            f"(task_id={message.task_id}, success={message.success})"
        )

        # Find task in registry
        task = await stream_registry.find_task_by_operation_id(message.operation_id)
        if task is None:
            task = await stream_registry.get_task(message.task_id)

        if task is None:
            logger.warning(
                f"[COMPLETION] Task not found for operation {message.operation_id} "
                f"(task_id={message.task_id})"
            )
            return

        logger.info(
            f"[COMPLETION] Found task: task_id={task.task_id}, "
            f"session_id={task.session_id}, tool_call_id={task.tool_call_id}"
        )

        # Guard against empty task fields
        if not task.task_id or not task.session_id or not task.tool_call_id:
            logger.error(
                f"[COMPLETION] Task has empty critical fields! "
                f"task_id={task.task_id!r}, session_id={task.session_id!r}, "
                f"tool_call_id={task.tool_call_id!r}"
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

        # Update pending operation in database using our Prisma client
        result_str = (
            message.result
            if isinstance(message.result, str)
            else (
                orjson.dumps(message.result).decode("utf-8")
                if message.result
                else '{"status": "completed"}'
            )
        )
        try:
            prisma = await self._ensure_prisma()
            await prisma.chatmessage.update_many(
                where={
                    "sessionId": task.session_id,
                    "toolCallId": task.tool_call_id,
                },
                data={"content": result_str},
            )
            logger.info(
                f"[COMPLETION] Updated tool message for session {task.session_id}"
            )
        except Exception as e:
            logger.error(
                f"[COMPLETION] Failed to update tool message: {e}", exc_info=True
            )

        # Generate LLM continuation with streaming
        try:
            await chat_service._generate_llm_continuation_with_streaming(
                session_id=task.session_id,
                user_id=task.user_id,
                task_id=task.task_id,
            )
        except Exception as e:
            logger.error(
                f"[COMPLETION] Failed to generate LLM continuation: {e}",
                exc_info=True,
            )

        # Mark task as completed and release Redis lock
        await stream_registry.mark_task_completed(task.task_id, status="completed")
        try:
            await chat_service._mark_operation_completed(task.tool_call_id)
        except Exception as e:
            logger.error(f"[COMPLETION] Failed to mark operation completed: {e}")

        logger.info(
            f"[COMPLETION] Successfully processed completion for task {task.task_id}"
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
        await stream_registry.publish_chunk(task.task_id, StreamFinish())

        # Update pending operation with error using our Prisma client
        error_response = ErrorResponse(
            message=error_msg,
            error=message.error,
        )
        try:
            prisma = await self._ensure_prisma()
            await prisma.chatmessage.update_many(
                where={
                    "sessionId": task.session_id,
                    "toolCallId": task.tool_call_id,
                },
                data={"content": error_response.model_dump_json()},
            )
            logger.info(
                f"[COMPLETION] Updated tool message with error for session {task.session_id}"
            )
        except Exception as e:
            logger.error(
                f"[COMPLETION] Failed to update tool message: {e}", exc_info=True
            )

        # Mark task as failed and release Redis lock
        await stream_registry.mark_task_completed(task.task_id, status="failed")
        try:
            await chat_service._mark_operation_completed(task.tool_call_id)
        except Exception as e:
            logger.error(f"[COMPLETION] Failed to mark operation completed: {e}")

        logger.info(
            f"[COMPLETION] Processed failure for task {task.task_id}: {error_msg}"
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
    """Publish an operation completion message to Redis Streams.

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

    redis = await get_redis_async()
    await redis.xadd(
        COMPLETION_STREAM,
        {"data": message.model_dump_json()},
        maxlen=STREAM_MAX_LENGTH,
    )
    logger.info(f"Published completion for operation {operation_id}")
