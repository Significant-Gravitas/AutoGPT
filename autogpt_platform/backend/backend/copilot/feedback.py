"""Thumbs up/down and copy ratings on AutoPilot replies.

Postgres (``ChatMessageFeedback``) keeps every rating, named after what it
rates ("user-feedback": 1 = up, 0 = down; "copy": 1). Rating a reply again
overwrites its rating and keeps its id.

Each rating also goes to PostHog LLM analytics as an ``$ai_metric`` event on
the user (distinct id = user id), for every user regardless of cookie
consent, like the generations OpenRouter Broadcast sends there.
``$ai_session_id`` is the chat session id, the same value the chat's
``$ai_generation`` events carry, which is what joins a rating to the
generations of its chat; ``message_id`` names the reply. ``$ai_trace_id``
is the trace of the turn that wrote the reply (``ChatMessage.traceId``, see
``copilot.ai_spans``), the one its generations, tool spans and ``$ai_trace``
carry, so the rating attaches to that reply's trace; a reply written before
the column existed has none and attaches to the chat only. Rating again sends
another event; the latest per ``feedback_id`` is the rating that stands.
"""

from typing import Any, Literal

from prisma.models import ChatMessage as PrismaChatMessage
from pydantic import BaseModel, Field

from backend.copilot import feedback_db
from backend.util.posthog_client import capture
from backend.util.posthog_events import PostHogEvent

FeedbackScoreName = Literal["user-feedback", "copy"]
FeedbackScoreValue = Literal[0, 1]

MAX_FEEDBACK_COMMENT_CHARS = 2000

# Sequences are Postgres ``int4``: more digits than this could overflow the
# lookup, and no chat is that long.
_MAX_SEQUENCE_DIGITS = 9


class MessageFeedbackResponse(BaseModel):
    """A saved rating of an AutoPilot reply."""

    id: str = Field(
        description="The rating's id, which its PostHog event carries as "
        "``feedback_id``.",
    )


async def record_message_feedback(
    *,
    user_id: str,
    session_id: str,
    message_id: str,
    score_name: FeedbackScoreName,
    score_value: FeedbackScoreValue,
    comment: str | None,
    impersonated: bool = False,
) -> MessageFeedbackResponse | None:
    """Save a rating of one reply in *user_id*'s chat and send it to PostHog.

    *message_id* is the id the chat UI knows the reply by: ``<session>-seq-<N>``
    for a reply loaded from the chat history, or the message row's own id.
    *impersonated* marks a rating an admin gave while acting as the user.

    Returns ``None`` when the session is not the caller's or *message_id* is
    not a reply in it.
    """
    message = await _find_rated_message(user_id, session_id, message_id)
    if message is None:
        return None
    feedback_id = await feedback_db.upsert_message_feedback(
        user_id=user_id,
        session_id=session_id,
        message_id=message.id,
        score_name=score_name,
        score_value=score_value,
        comment=comment,
    )
    _send_to_posthog(
        user_id=user_id,
        feedback_id=feedback_id,
        session_id=session_id,
        message=message,
        score_name=score_name,
        score_value=score_value,
        comment=comment,
        impersonated=impersonated,
    )
    return MessageFeedbackResponse(id=feedback_id)


def sequence_from_ui_message_id(session_id: str, message_id: str) -> int | None:
    """The row sequence a chat-UI message id names, if it names one.

    The copilot UI keys a reply loaded from history ``<session>-seq-<N>``,
    N being the sequence of the last row it folded into the bubble (see
    ``convertChatSessionToUiMessages`` in the frontend).
    """
    digits = message_id.removeprefix(f"{session_id}-seq-")
    if digits == message_id or not (digits.isascii() and digits.isdigit()):
        return None
    if len(digits) > _MAX_SEQUENCE_DIGITS:
        return None
    return int(digits)


async def _find_rated_message(
    user_id: str, session_id: str, message_id: str
) -> PrismaChatMessage | None:
    sequence = sequence_from_ui_message_id(session_id, message_id)
    if sequence is not None:
        return await feedback_db.get_rateable_message(
            user_id, session_id, sequence=sequence
        )
    return await feedback_db.get_rateable_message(
        user_id, session_id, message_id=message_id
    )


def _send_to_posthog(
    *,
    user_id: str,
    feedback_id: str,
    session_id: str,
    message: PrismaChatMessage,
    score_name: FeedbackScoreName,
    score_value: FeedbackScoreValue,
    comment: str | None,
    impersonated: bool,
) -> None:
    """Queue the rating as an ``$ai_metric`` event. ``capture`` logs and
    swallows a PostHog failure, so the rating is never lost to one."""
    properties: dict[str, Any] = {
        "$ai_metric_name": score_name,
        "$ai_metric_value": score_value,
        "$ai_session_id": session_id,
        "chat_session_id": session_id,
        "message_id": message.id,
        "message_sequence": message.sequence,
        "feedback_id": feedback_id,
        "impersonated": impersonated,
    }
    if message.traceId:
        properties["$ai_trace_id"] = message.traceId
    if comment:
        properties["$ai_feedback_text"] = comment
    capture(user_id, PostHogEvent.AI_METRIC, properties)
