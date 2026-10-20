import logging
from typing import TYPE_CHECKING

from langfuse import get_client

if TYPE_CHECKING:
    from backend.copilot.pending_messages import PendingMessage

logger = logging.getLogger(__name__)


def record_injected_messages(
    messages: list["PendingMessage"],
    *,
    injected_contents: list[str],
    source: str,
    tool_use_id: str | None,
    tool_name: str | None,
) -> None:
    """Trace only the rendered prefix of messages supplied to the model."""
    try:
        client = get_client()
        if client.get_current_trace_id() is None:
            return
        for index, (message, content) in enumerate(
            zip(messages, injected_contents), start=1
        ):
            client.create_event(
                name="copilot-user-follow-up",
                input=[{"role": "user", "content": content}],
                metadata={
                    "source": source,
                    "message_id": message.id,
                    "enqueued_at": message.enqueued_at,
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "batch_index": index,
                    "excluded_message_count": len(messages) - len(injected_contents),
                    "file_ids": message.file_ids,
                    "context": (
                        {"url": message.context.url}
                        if message.context and message.context.url
                        else None
                    ),
                },
            )
    except Exception:
        logger.debug("Failed to trace injected user messages", exc_info=True)
