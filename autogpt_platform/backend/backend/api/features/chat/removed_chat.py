"""The answer for a chat that resetting its expert removed (SECRT-2692).

The rows stay in the database, but the chat is never opened or continued
again. Old links still reach it from Otto's cards and Home, so opening one
answers 410 with a typed detail the chat page recognises, instead of the 404
a chat that never existed gets.
"""

from typing import Literal

import prisma.models
from fastapi import HTTPException, status
from pydantic import BaseModel

from backend.copilot.model import ChatSessionInfo


class ChatRemovedDetail(BaseModel):
    """Why the chat cannot be opened, and whose reset removed it."""

    type: Literal["chat_removed"] = "chat_removed"
    expert_name: str | None = None
    message: str


class ChatRemovedResponse(BaseModel):
    """The 410 body; FastAPI wraps an HTTPException's detail like this."""

    detail: ChatRemovedDetail


CHAT_REMOVED_RESPONSE = {
    status.HTTP_410_GONE: {
        "model": ChatRemovedResponse,
        "description": "The chat was removed when its expert was reset",
    }
}


async def chat_removed_error(session: ChatSessionInfo) -> HTTPException:
    """The 410 for *session*, which a reset archived, naming the expert."""
    name = None
    if session.expert_id is not None:
        expert = await prisma.models.Expert.prisma().find_first(
            where={"id": session.expert_id, "ownerUserId": session.user_id}
        )
        name = expert.name if expert else None
    message = (
        f"This chat was removed when {name} was reset."
        if name
        else "This chat was removed when its expert was reset."
    )
    return HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=ChatRemovedDetail(expert_name=name, message=message).model_dump(),
    )
