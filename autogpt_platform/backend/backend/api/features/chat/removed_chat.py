"""The answer for a chat that resetting its expert removed (SECRT-2692).

The rows stay in the database, but the chat is never opened or continued
again. Old links still reach it from Otto's cards and Home, so opening one
answers 410 with a typed detail the chat page recognises, instead of the 404
a chat that never existed gets.
"""

from typing import Any, Literal

import prisma.models
from fastapi import HTTPException, status
from pydantic import BaseModel

from backend.copilot.model import ChatSessionInfo


class ChatRemovedDetail(BaseModel):
    """Why the chat cannot be opened, and whose reset removed it."""

    type: Literal["chat_removed"] = "chat_removed"
    # Set only while the expert is still on the team, so the page can offer a
    # chat with them; a fired expert is named but not linked.
    expert_id: str | None = None
    expert_name: str | None = None
    message: str


class ChatRemovedResponse(BaseModel):
    """The 410 body; FastAPI wraps an HTTPException's detail like this."""

    detail: ChatRemovedDetail


CHAT_REMOVED_RESPONSE: dict[int | str, dict[str, Any]] = {
    status.HTTP_410_GONE: {
        "model": ChatRemovedResponse,
        "description": "The chat was removed when its expert was reset",
    }
}


async def chat_removed_error(session: ChatSessionInfo) -> HTTPException:
    """The 410 for *session*, which a reset archived, naming the expert."""
    name = None
    expert_id = None
    if session.expert_id is not None:
        expert = await prisma.models.Expert.prisma().find_first(
            where={"id": session.expert_id, "ownerUserId": session.user_id}
        )
        if expert is not None:
            name = expert.name
            expert_id = None if expert.isArchived else expert.id
    message = (
        f"This chat was removed when {name} was reset."
        if name
        else "This chat was removed when its expert was reset."
    )
    return HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=ChatRemovedDetail(
            expert_id=expert_id, expert_name=name, message=message
        ).model_dump(),
    )
