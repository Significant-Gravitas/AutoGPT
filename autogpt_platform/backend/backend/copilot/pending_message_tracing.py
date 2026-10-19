import logging
from typing import TYPE_CHECKING

from langfuse import get_client

if TYPE_CHECKING:
    from backend.copilot.pending_messages import PendingMessage

logger = logging.getLogger(__name__)


def record_injected_messages(
    messages: list["PendingMessage"],
    *,
    tool_use_id: str | None,
    tool_name: str | None,
) -> None:
    """Record user input at the tool boundary that consumed it, within its turn."""
    try:
        client = get_client()
        if client.get_current_trace_id() is None:
            return
        for index, message in enumerate(messages, start=1):
            client.create_event(
                name="copilot-user-follow-up",
                input=[{"role": "user", "content": message.content}],
                metadata={
                    "source": "PostToolUse",
                    "message_id": message.id,
                    "enqueued_at": message.enqueued_at,
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "batch_index": index,
                    "file_ids": message.file_ids,
                    "context": (
                        message.context.model_dump() if message.context else None
                    ),
                },
            )
    except Exception:
        logger.debug("Failed to trace injected user messages", exc_info=True)
