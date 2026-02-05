"""Redis Streams consumer for operation completion messages.

This module provides a consumer (ChatCompletionConsumer) that listens for
completion notifications (OperationCompleteMessage) from external services
(like Agent Generator) and triggers the appropriate stream registry and
chat service updates via process_operation_success/process_operation_failure.

Why Redis Streams instead of RabbitMQ?
--------------------------------------
While the project typically uses RabbitMQ for async task queues (e.g., execution
queue), Redis Streams was chosen for chat completion notifications because:

1. **Unified Infrastructure**: The SSE reconnection feature already uses Redis
   Streams (via stream_registry) for message persistence and replay. Using Redis
   Streams for completion notifications keeps all chat streaming infrastructure
   in one system, simplifying operations and reducing cross-system coordination.

2. **Message Replay**: Redis Streams support XREAD with arbitrary message IDs,
   allowing consumers to replay missed messages after reconnection. This aligns
   with the SSE reconnection pattern where clients can resume from last_message_id.

3. **Consumer Groups with XAUTOCLAIM**: Redis consumer groups provide automatic
   load balancing across pods with explicit message claiming (XAUTOCLAIM) for
   recovering from dead consumers - ideal for the completion callback pattern.

4. **Lower Latency**: For real-time SSE updates, Redis (already in-memory for
   stream_registry) provides lower latency than an additional RabbitMQ hop.

5. **Atomicity with Task State**: Completion processing often needs to update
   task metadata stored in Redis. Keeping both in Redis enables simpler
   transactional semantics without distributed coordination.

The consumer uses Redis Streams with consumer groups for reliable message
processing across multiple platform pods, with XAUTOCLAIM for reclaiming
stale pending messages from dead consumers.
"""

import asyncio
import logging
import os
import uuid
from typing import Any

import orjson
from prisma import Prisma
from pydantic import BaseModel
from redis.exceptions import ResponseError

from backend.data.redis_client import get_redis_async

from . import stream_registry
from .completion_handler import process_operation_failure, process_operation_success
from .config import ChatConfig

logger = logging.getLogger(__name__)
config = ChatConfig()


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
                config.stream_completion_name,
                config.stream_consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                f"Created consumer group '{config.stream_consumer_group}' "
                f"on stream '{config.stream_completion_name}'"
            )
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(
                    f"Consumer group '{config.stream_consumer_group}' already exists"
                )
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
                    # First, claim any stale pending messages from dead consumers
                    # Redis does NOT auto-redeliver pending messages; we must explicitly
                    # claim them using XAUTOCLAIM
                    try:
                        claimed_result = await redis.xautoclaim(
                            name=config.stream_completion_name,
                            groupname=config.stream_consumer_group,
                            consumername=self._consumer_name,
                            min_idle_time=config.stream_claim_min_idle_ms,
                            start_id="0-0",
                            count=10,
                        )
                        # xautoclaim returns: (next_start_id, [(id, data), ...], [deleted_ids])
                        if claimed_result and len(claimed_result) >= 2:
                            claimed_entries = claimed_result[1]
                            if claimed_entries:
                                logger.info(
                                    f"Claimed {len(claimed_entries)} stale pending messages"
                                )
                                for entry_id, data in claimed_entries:
                                    if not self._running:
                                        return
                                    await self._process_entry(redis, entry_id, data)
                    except Exception as e:
                        logger.warning(f"XAUTOCLAIM failed (non-fatal): {e}")

                    # Read new messages from the stream
                    messages = await redis.xreadgroup(
                        groupname=config.stream_consumer_group,
                        consumername=self._consumer_name,
                        streams={config.stream_completion_name: ">"},
                        block=block_timeout,
                        count=10,
                    )

                    if not messages:
                        continue

                    for stream_name, entries in messages:
                        for entry_id, data in entries:
                            if not self._running:
                                return
                            await self._process_entry(redis, entry_id, data)

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

    async def _process_entry(
        self, redis: Any, entry_id: str, data: dict[str, Any]
    ) -> None:
        """Process a single stream entry and acknowledge it on success.

        Args:
            redis: Redis client connection
            entry_id: The stream entry ID
            data: The entry data dict
        """
        try:
            # Handle the message
            message_data = data.get("data")
            if message_data:
                await self._handle_message(
                    message_data.encode()
                    if isinstance(message_data, str)
                    else message_data
                )

            # Acknowledge the message after successful processing
            await redis.xack(
                config.stream_completion_name,
                config.stream_consumer_group,
                entry_id,
            )
        except Exception as e:
            logger.error(
                f"Error processing completion message {entry_id}: {e}",
                exc_info=True,
            )
            # Message remains in pending state and will be claimed by
            # XAUTOCLAIM after min_idle_time expires

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
        prisma = await self._ensure_prisma()
        await process_operation_success(task, message.result, prisma)

    async def _handle_failure(
        self,
        task: stream_registry.ActiveTask,
        message: OperationCompleteMessage,
    ) -> None:
        """Handle failed operation completion."""
        prisma = await self._ensure_prisma()
        await process_operation_failure(task, message.error, prisma)


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
        config.stream_completion_name,
        {"data": message.model_dump_json()},
        maxlen=config.stream_max_length,
    )
    logger.info(f"Published completion for operation {operation_id}")
