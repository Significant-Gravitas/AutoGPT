"""Database access for ratings on AutoPilot replies.

Every query is scoped to the caller: a message is only found through a chat
session ``user_id`` owns, and a rating row is keyed by ``user_id``, so a
caller can neither read nor rate another user's replies even if a route
forgot its own ownership check.
"""

from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatMessageFeedback
from prisma.types import (
    ChatMessageFeedbackCreateInput,
    ChatMessageFeedbackUpdateInput,
    ChatMessageFeedbackUpsertInput,
    ChatMessageFeedbackWhereUniqueInput,
    ChatMessageWhereInput,
)

from backend.util.json import sanitize_string

# The rows a reply bubble in the chat UI can be keyed by: the UI folds a
# turn's assistant and reasoning rows into one bubble and names it after the
# last of them. User, tool and system rows are never rated.
RATEABLE_ROLES = ("assistant", "reasoning")


async def get_rateable_message(
    user_id: str,
    session_id: str,
    *,
    message_id: str | None = None,
    sequence: int | None = None,
) -> PrismaChatMessage | None:
    """The reply row in *user_id*'s session, found by id or by sequence.

    Returns ``None`` when the session is not the caller's, when no such row
    exists in it, or when the row is not a reply.
    """
    if (message_id is None) == (sequence is None):
        raise ValueError("Pass exactly one of message_id and sequence")
    where: ChatMessageWhereInput = {
        "sessionId": session_id,
        "Session": {"is": {"userId": user_id}},
        "role": {"in": list(RATEABLE_ROLES)},
    }
    if message_id is not None:
        where["id"] = message_id
    if sequence is not None:
        where["sequence"] = sequence
    return await PrismaChatMessage.prisma().find_first(where=where)


async def upsert_message_feedback(
    *,
    user_id: str,
    session_id: str,
    message_id: str,
    score_name: str,
    score_value: int,
    comment: str | None,
    langfuse_trace_id: str | None,
) -> str:
    """Save *user_id*'s rating of a reply and return the rating's id.

    One row per user, message and score name: rating again overwrites the
    value and comment and keeps the id, which is also the Langfuse score id.
    The caller must already have checked that the message is in a session
    *user_id* owns (see ``get_rateable_message``).
    """
    comment = sanitize_string(comment) if comment is not None else None
    where: ChatMessageFeedbackWhereUniqueInput = {
        "messageId_userId_scoreName": {
            "messageId": message_id,
            "userId": user_id,
            "scoreName": score_name,
        }
    }
    create: ChatMessageFeedbackCreateInput = {
        "userId": user_id,
        "messageId": message_id,
        "sessionId": session_id,
        "scoreName": score_name,
        "scoreValue": score_value,
        "comment": comment,
        "langfuseTraceId": langfuse_trace_id,
    }
    update: ChatMessageFeedbackUpdateInput = {
        "scoreValue": score_value,
        "comment": comment,
        "langfuseTraceId": langfuse_trace_id,
    }
    data: ChatMessageFeedbackUpsertInput = {"create": create, "update": update}
    try:
        row = await ChatMessageFeedback.prisma().upsert(where=where, data=data)
    except UniqueViolationError:
        # Two clicks raced on the first rating of this reply and the other
        # one inserted the row between this upsert's read and its insert.
        # It exists now, so the retry updates it.
        row = await ChatMessageFeedback.prisma().upsert(where=where, data=data)
    return row.id
