"""Error markers persisted into a chat so a failure survives the stream.

A stream error lives only on the wire. When the SSE connection ends the
message is gone, and a reloaded chat shows the user's question followed by
nothing at all -- no answer, no error, no sign anything happened. The SDK
path solved this by appending a marker row; the baseline path never did.

The prefixes are a frontend rendering contract, not decoration:
``COPILOT_ERROR_PREFIX`` renders an error card, and
``COPILOT_RETRYABLE_ERROR_PREFIX`` renders one that also offers Try Again.
Choosing between them is a claim about whether trying again can work, so it
is made from the typed failure rather than assumed.
"""

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    COPILOT_SYSTEM_PREFIX,
    STREAM_ERROR_MARKER,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.model import ChatMessage, ChatSession


def is_error_marker(message: ChatMessage) -> bool:
    """True when this row is a marker rather than something the model said."""
    if message.role != "assistant" or not message.content:
        return False
    return message.content.startswith(
        (COPILOT_ERROR_PREFIX, COPILOT_RETRYABLE_ERROR_PREFIX)
    )


def has_trailing_marker(session: ChatSession | None) -> bool:
    """True when the last row is already a marker.

    Several guards can fire on one failed turn. Without this check a single
    failure stacks two or three error cards on the same chat.
    """
    if session is None or not session.messages:
        return False
    last = session.messages[-1]
    if last.role != "assistant" or not last.content:
        return False
    return (
        is_error_marker(last)
        or last.content.startswith(COPILOT_SYSTEM_PREFIX)
        or last.content in (STREAM_ERROR_MARKER, STREAM_INCOMPLETE_MARKER)
    )


def append_error_marker(
    session: ChatSession | None,
    display_message: str,
    *,
    retryable: bool,
) -> bool:
    """Record a failure in the chat itself. Returns whether a row was added.

    ``retryable`` decides which card the user sees, so it must come from
    something that actually knows -- offering Try Again on an expired login
    or a spent quota costs the user a retry to learn it cannot work.
    """
    if session is None or has_trailing_marker(session):
        return False
    prefix = COPILOT_RETRYABLE_ERROR_PREFIX if retryable else COPILOT_ERROR_PREFIX
    session.messages.append(
        ChatMessage(role="assistant", content=f"{prefix} {display_message}")
    )
    return True
