"""RabbitMQ queue configuration for CoPilot executor.

Defines two exchanges and queues following the graph executor pattern:
- 'copilot_execution' (DIRECT) for chat generation tasks
- 'copilot_cancel' (FANOUT) for cancellation requests
"""

import logging

from pydantic import BaseModel

from backend.copilot.config import CopilotLlmModel, CopilotMode
from backend.copilot.permissions import CopilotPermissions
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
# ``_v2`` suffix marks the classic→quorum rollover; old-image consumers
# drain the unsuffixed queue. Orphans cleaned up in a follow-up PR.
COPILOT_EXECUTION_QUEUE_NAME = "copilot_execution_queue_v2"
COPILOT_EXECUTION_ROUTING_KEY = "copilot.run"

COPILOT_CANCEL_EXCHANGE = Exchange(
    name="copilot_cancel",
    type=ExchangeType.FANOUT,
    durable=True,
    auto_delete=False,
)
COPILOT_CANCEL_QUEUE_NAME = "copilot_cancel_queue_v2"


def get_session_lock_key(session_id: str) -> str:
    """Redis key for the per-session cluster lock held by the executing pod."""
    return f"copilot:session:{session_id}:lock"


# CoPilot operations can include extended thinking and agent generation
# which may take several hours to complete. Matches the pod's
# terminationGracePeriodSeconds in the helm chart so a rolling deploy can let
# the longest legitimate turn finish. Also bounds the stale-session auto-
# complete watchdog in stream_registry (consumer_timeout + 5min buffer).
COPILOT_CONSUMER_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours

# Graceful shutdown timeout - must match COPILOT_CONSUMER_TIMEOUT_SECONDS so
# cleanup can let the longest legitimate turn complete before the pod is
# SIGKILL'd by kubelet.
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = COPILOT_CONSUMER_TIMEOUT_SECONDS


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
            # Quorum (not classic mirrored) for leader election + stronger
            # replication across RabbitMQ 4.x cluster nodes.
            "x-queue-type": "quorum",
            # Consumer timeout matches the pod graceful-shutdown window so a
            # rolling deploy never forces redelivery of a turn that the pod
            # is still legitimately finishing.
            #
            # Deploy note: RabbitMQ (verified on 4.1.4) does NOT strictly
            # compare ``x-consumer-timeout`` on queue redeclaration, so this
            # value can change between deploys without triggering
            # PRECONDITION_FAILED. To update the *effective* timeout on an
            # already-running queue before the new code deploys (so pods
            # mid-shutdown don't have their consumer cancelled at the old
            # limit), apply a policy:
            #
            #     rabbitmqctl set_policy copilot-consumer-timeout \
            #       "^copilot_execution_queue_v2$" \
            #       '{"consumer-timeout": 21600000}' \
            #       --apply-to queues
            #
            # The policy takes effect immediately. Once the policy is set
            # to match the code's value the policy is redundant for new
            # pods and can be removed after a stable deploy if desired —
            # but it's harmless to leave in place.
            "x-consumer-timeout": COPILOT_CONSUMER_TIMEOUT_SECONDS * 1000,
        },
    )
    cancel_queue = Queue(
        name=COPILOT_CANCEL_QUEUE_NAME,
        exchange=COPILOT_CANCEL_EXCHANGE,
        routing_key="",  # not used for FANOUT
        durable=True,
        auto_delete=False,
        arguments={"x-queue-type": "quorum"},
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

    file_ids: list[str] | None = None
    """Workspace file IDs attached to the user's message"""

    mode: CopilotMode | None = None
    """Autopilot mode override: 'fast' or 'extended_thinking'. None = server default."""

    model: CopilotLlmModel | None = None
    """Per-request model tier: 'standard' or 'advanced'. None = server default."""

    permissions: CopilotPermissions | None = None
    """Capability filter inherited from a parent run (e.g. ``run_sub_session``
    forwards its parent's permissions so the sub can't escalate). ``None``
    means the worker applies no filter."""

    request_arrival_at: float = 0.0
    """Unix-epoch seconds (server clock) when the originating HTTP
    ``/stream`` request arrived.  The executor's turn-start drain uses
    this to decide whether each pending message was typed BEFORE or AFTER
    the turn's ``current`` message, and orders the combined user bubble
    chronologically.  Defaults to ``0.0`` for backward compatibility with
    queue messages written before this field existed (they sort as "all
    pending before current" — the pre-fix behaviour)."""


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
    file_ids: list[str] | None = None,
    mode: CopilotMode | None = None,
    model: CopilotLlmModel | None = None,
    permissions: CopilotPermissions | None = None,
    request_arrival_at: float = 0.0,
) -> None:
    """Enqueue a CoPilot task for processing by the executor service.

    Args:
        session_id: Chat session ID (also used for dedup/locking)
        user_id: User ID (may be None for anonymous users)
        message: User's message to process
        turn_id: Per-turn UUID for Redis stream isolation
        is_user_message: Whether the message is from the user (vs system/assistant)
        context: Optional context for the message (e.g., {url: str, content: str})
        file_ids: Optional workspace file IDs attached to the user's message
        mode: Autopilot mode override ('fast' or 'extended_thinking'). None = server default.
        model: Per-request model tier ('standard' or 'advanced'). None = server default.
        permissions: Capability filter inherited from a parent run (sub-AutoPilot).
            None = no filter.
    """
    from backend.util.clients import get_async_copilot_queue

    entry = CoPilotExecutionEntry(
        session_id=session_id,
        turn_id=turn_id,
        user_id=user_id,
        message=message,
        is_user_message=is_user_message,
        context=context,
        file_ids=file_ids,
        mode=mode,
        model=model,
        permissions=permissions,
        request_arrival_at=request_arrival_at,
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
