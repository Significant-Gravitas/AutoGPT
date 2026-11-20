"""Classify chat replies for first-output and first-answer measurements."""

import logging
from typing import TYPE_CHECKING

from sentry_sdk.tracing import Span

from backend.copilot.constants import is_platform_text
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputStart,
)
from backend.copilot.service import strip_user_context_tags

if TYPE_CHECKING:
    from backend.copilot.model import ChatMessage, ChatSession

logger = logging.getLogger(__name__)
MODEL_OUTPUT_CHUNKS = (StreamTextDelta, StreamReasoningDelta, StreamToolInputStart)


def describe_chat(
    span: Span, session: "ChatSession", message: str | None = None
) -> None:
    """Tag the kind of chat the turn is in, so its timings split by it: a
    chat's first turn sets up its sandbox and context, and an expert's
    turn loads the expert as well. The first turn answers the chat's first
    user message; a greeting or note the platform wrote before it doesn't
    make it a follow-up.

    ``message`` is the user message the turn answers, its server tags
    stripped. The chat route saves it before dispatch, sometimes behind a
    server block (a voice turn's instructions, a first turn's context); a
    sub-session doesn't, and the engine saves it when the turn starts.
    Either way it counts once, and so does a message of files with no
    text, which nothing saves. A turn with no user message of its own (a
    kickoff, a scheduled follow-up the scheduler stores as the model's, a
    continuation) is the first only in a chat nobody has written to yet.
    A chat loaded without its oldest messages (a long one, past the load
    limit) has had turns before the ones it holds."""
    try:
        users = sum(1 for row in session.messages if row.role == "user")
        if message is None:
            first = users == 0
        else:
            if not _is_latest_user_message(session.messages, message):
                users += 1
            first = users <= 1
        if session.messages and (session.messages[0].sequence or 0) > 0:
            first = False
        span.set_tag("copilot.turn.first", "true" if first else "false")
        span.set_tag("copilot.expert", "true" if session.expert_id else "false")
    except Exception:
        logger.debug("Could not tag the turn's kind of chat", exc_info=True)


def is_model_output(chunk: StreamBaseResponse) -> bool:
    """Whether the chunk carries model output; a notice the platform streams
    as text, where the model gave no answer, does not."""
    return isinstance(chunk, MODEL_OUTPUT_CHUNKS) and not _is_platform_delta(chunk)


def is_answer_text(chunk: StreamBaseResponse) -> bool:
    """Whether the chunk is text of the model's answer. Whitespace on its own
    shows nothing, so it doesn't start the answer."""
    return (
        isinstance(chunk, StreamTextDelta)
        and bool(chunk.delta.strip())
        and not is_platform_text(chunk.delta)
    )


def _is_platform_delta(chunk: StreamBaseResponse) -> bool:
    return isinstance(chunk, StreamTextDelta) and is_platform_text(chunk.delta)


def _is_latest_user_message(messages: list["ChatMessage"], message: str) -> bool:
    """Whether *message* is already saved at the end of the chat, compared
    without the server blocks a saved copy can carry. Stripping a block also
    takes the whitespace after it, so leading whitespace is ignored on both
    sides."""
    for row in reversed(messages):
        if row.role != "user":
            return False
        if strip_user_context_tags(row.content or "").lstrip() == message.lstrip():
            return True
    return False
