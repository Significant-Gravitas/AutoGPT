"""RabbitMQ queue configuration for CoPilot executor.

Defines two exchanges and queues following the graph executor pattern:
- 'copilot_execution' (DIRECT) for chat generation tasks
- 'copilot_cancel' (FANOUT) for cancellation requests
"""

import logging

from pydantic import BaseModel

from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.util.logging import TruncatedLogger, is_structured_logging_enabled

logger = logging.getLogger(__name__)


# ============ Logging Helper ============ #


class CoPilotLogMetadata(TruncatedLogger):
    """Structured logging helper for CoPilot executor.

    In cloud environments (structured logging enabled), uses a simple prefix
    and passes metadata via json_fields. In local environments, uses a detailed
    prefix with all metadata key-value pairs for easier debugging.

    Args:
        logger: The underlying logger instance
        max_length: Maximum log message length before truncation
        **kwargs: Metadata key-value pairs (e.g., session_id="xyz", turn_id="abc")
            These are added to json_fields in cloud mode, or to the prefix in local mode.
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_length: int = 1000,
        **kwargs: str | None,
    ):
        # Filter out None values
        metadata = {k: v for k, v in kwargs.items() if v is not None}
        metadata["component"] = "CoPilotExecutor"

        if is_structured_logging_enabled():
            prefix = "[CoPilotExecutor]"
        else:
            # Build prefix from metadata key-value pairs
            meta_parts = "|".join(
                f"{k}:{v}" for k, v in metadata.items() if k != "component"
            )
            prefix = (
                f"[CoPilotExecutor|{meta_parts}]" if meta_parts else "[CoPilotExecutor]"
            )

        super().__init__(
            logger,
            max_length=max_length,
            prefix=prefix,
            metadata=metadata,
        )


# ============ Exchange and Queue Configuration ============ #

COPILOT_EXECUTION_EXCHANGE = Exchange(
    name="copilot_execution",
    type=ExchangeType.DIRECT,
    durable=True,
    auto_delete=False,
)
COPILOT_EXECUTION_QUEUE_NAME = "copilot_execution_queue"
COPILOT_EXECUTION_ROUTING_KEY = "copilot.run"

COPILOT_CANCEL_EXCHANGE = Exchange(
    name="copilot_cancel",
    type=ExchangeType.FANOUT,
    durable=True,
    auto_delete=False,
)
COPILOT_CANCEL_QUEUE_NAME = "copilot_cancel_queue"

# CoPilot operations can include extended thinking and agent generation
# which may take 30+ minutes to complete
COPILOT_CONSUMER_TIMEOUT_SECONDS = 60 * 60  # 1 hour

# Graceful shutdown timeout - allow in-flight operations to complete
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 30 * 60  # 30 minutes


def create_copilot_queue_config() -> RabbitMQConfig:
    """Create RabbitMQ configuration for CoPilot executor.

    Defines two exchanges and queues:
    - 'copilot_execution' (DIRECT) for chat generation tasks
    - 'copilot_cancel' (FANOUT) for cancellation requests

    Returns:
        RabbitMQConfig with exchanges and queues defined
    """
    run_queue = Queue(
        name=COPILOT_EXECUTION_QUEUE_NAME,
        exchange=COPILOT_EXECUTION_EXCHANGE,
        routing_key=COPILOT_EXECUTION_ROUTING_KEY,
        durable=True,
        auto_delete=False,
        arguments={
            # Extended consumer timeout for long-running LLM operations
            # Default 30-minute timeout is insufficient for extended thinking
            # and agent generation which can take 30+ minutes
            "x-consumer-timeout": COPILOT_CONSUMER_TIMEOUT_SECONDS
            * 1000,
        },
    )
    cancel_queue = Queue(
        name=COPILOT_CANCEL_QUEUE_NAME,
        exchange=COPILOT_CANCEL_EXCHANGE,
        routing_key="",  # not used for FANOUT
        durable=True,
        auto_delete=False,
    )
    return RabbitMQConfig(
        vhost="/",
        exchanges=[COPILOT_EXECUTION_EXCHANGE, COPILOT_CANCEL_EXCHANGE],
        queues=[run_queue, cancel_queue],
    )


# ============ Message Models ============ #


class CoPilotExecutionEntry(BaseModel):
    """Task payload for CoPilot AI generation.

    This model represents a chat generation task to be processed by the executor.
    """

    session_id: str
    """Chat session ID (also used for dedup/locking)"""

    turn_id: str = ""
    """Per-turn UUID for Redis stream isolation"""

    user_id: str | None
    """User ID (may be None for anonymous users)"""

    message: str
    """User's message to process"""

    is_user_message: bool = True
    """Whether the message is from the user (vs system/assistant)"""

    context: dict[str, str] | None = None
    """Optional context for the message (e.g., {url: str, content: str})"""


class CancelCoPilotEvent(BaseModel):
    """Event to cancel a CoPilot operation."""

    session_id: str
    """Session ID to cancel"""


# ============ Queue Publishing Helpers ============ #


async def enqueue_copilot_turn(
    session_id: str,
    user_id: str | None,
    message: str,
    turn_id: str,
    is_user_message: bool = True,
    context: dict[str, str] | None = None,
) -> None:
    """Enqueue a CoPilot task for processing by the executor service.

    Args:
        session_id: Chat session ID (also used for dedup/locking)
        user_id: User ID (may be None for anonymous users)
        message: User's message to process
        turn_id: Per-turn UUID for Redis stream isolation
        is_user_message: Whether the message is from the user (vs system/assistant)
        context: Optional context for the message (e.g., {url: str, content: str})
    """
    from backend.util.clients import get_async_copilot_queue

    entry = CoPilotExecutionEntry(
        session_id=session_id,
        turn_id=turn_id,
        user_id=user_id,
        message=message,
        is_user_message=is_user_message,
        context=context,
    )

    queue_client = await get_async_copilot_queue()
    await queue_client.publish_message(
        routing_key=COPILOT_EXECUTION_ROUTING_KEY,
        message=entry.model_dump_json(),
        exchange=COPILOT_EXECUTION_EXCHANGE,
    )


async def enqueue_cancel_task(session_id: str) -> None:
    """Publish a cancel request for a running CoPilot session.

    Sends a ``CancelCoPilotEvent`` to the FANOUT exchange so all executor
    pods receive the cancellation signal.
    """
    from backend.util.clients import get_async_copilot_queue

    event = CancelCoPilotEvent(session_id=session_id)
    queue_client = await get_async_copilot_queue()
    await queue_client.publish_message(
        routing_key="",  # FANOUT ignores routing key
        message=event.model_dump_json(),
        exchange=COPILOT_CANCEL_EXCHANGE,
    )
