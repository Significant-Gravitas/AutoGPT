"""Thumbs up/down and copy ratings on AutoPilot replies.

Postgres (``ChatMessageFeedback``) keeps every rating. Each one is also sent
to Langfuse as a score named after the rating ("user-feedback": 1 = up,
0 = down; "copy": 1), in the environment the copilot traces use:

- on the trace of the SDK turn that wrote the reply, which the turn stamps
  on its assistant rows (``ChatMessage.langfuseTraceId``);
- otherwise on the chat's Langfuse session, with the message in the score's
  metadata: baseline turns have no turn-level trace, and replies written
  before the stamp existed have none recorded.

The score reuses the rating's id, so rating a reply again updates its score
instead of adding one. A Langfuse failure is logged and shows in the
response; the rating itself is already saved.
"""

import logging
from typing import Literal

from langfuse import get_client
from langfuse.api import CreateScoreRequest, ScoreDataType
from prisma.models import ChatMessage as PrismaChatMessage
from pydantic import BaseModel, Field

from backend.copilot import feedback_db
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

FeedbackScoreName = Literal["user-feedback", "copy"]
FeedbackScoreValue = Literal[0, 1]
LangfuseScoreTarget = Literal["trace", "session"]

MAX_FEEDBACK_COMMENT_CHARS = 2000

# The rating waits on Langfuse, so an unresponsive Langfuse must not hold the
# click for long; the rating is saved regardless.
_LANGFUSE_TIMEOUT_SECONDS = 5

# Sequences are Postgres ``int4``: more digits than this could overflow the
# lookup, and no chat is that long.
_MAX_SEQUENCE_DIGITS = 9


class MessageFeedbackResponse(BaseModel):
    """A saved rating of an AutoPilot reply."""

    id: str = Field(
        description="The rating's id, which its Langfuse score shares.",
    )
    langfuse_target: LangfuseScoreTarget | None = Field(
        default=None,
        description=(
            "Where the Langfuse score went: the trace of the turn that wrote "
            "the reply, or the chat's Langfuse session when that trace is not "
            "known. Null when Langfuse is not configured or did not take the "
            "score; the rating is saved either way."
        ),
    )


async def record_message_feedback(
    *,
    user_id: str,
    session_id: str,
    message_id: str,
    score_name: FeedbackScoreName,
    score_value: FeedbackScoreValue,
    comment: str | None,
) -> MessageFeedbackResponse | None:
    """Save a rating of one reply in *user_id*'s chat and score it in Langfuse.

    *message_id* is the id the chat UI knows the reply by: ``<session>-seq-<N>``
    for a reply loaded from the chat history, or the message row's own id.

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
        langfuse_trace_id=message.langfuseTraceId,
    )
    target = await _score_in_langfuse(
        feedback_id=feedback_id,
        session_id=session_id,
        message=message,
        score_name=score_name,
        score_value=score_value,
        comment=comment,
    )
    return MessageFeedbackResponse(id=feedback_id, langfuse_target=target)


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


async def _score_in_langfuse(
    *,
    feedback_id: str,
    session_id: str,
    message: PrismaChatMessage,
    score_name: FeedbackScoreName,
    score_value: FeedbackScoreValue,
    comment: str | None,
) -> LangfuseScoreTarget | None:
    """Send the rating to Langfuse and say where it landed, or None."""
    if not (
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    ):
        return None
    trace_id = message.langfuseTraceId
    request = CreateScoreRequest(
        id=feedback_id,
        # A score belongs to a trace or to a session, not both.
        traceId=trace_id,
        sessionId=None if trace_id else session_id,
        name=score_name,
        value=score_value,
        dataType=ScoreDataType.NUMERIC,
        comment=comment,
        metadata={
            "message_id": message.id,
            "message_sequence": message.sequence,
            "session_id": session_id,
        },
        environment=settings.secrets.langfuse_tracing_environment,
    )
    try:
        await get_client().async_api.score.create(
            request=request,
            request_options={"timeout_in_seconds": _LANGFUSE_TIMEOUT_SECONDS},
        )
    except Exception:
        logger.warning(
            f"Langfuse did not take {score_name} score {feedback_id} for "
            f"session {session_id}; the rating is saved in Postgres only",
            exc_info=True,
        )
        return None
    return "trace" if trace_id else "session"
