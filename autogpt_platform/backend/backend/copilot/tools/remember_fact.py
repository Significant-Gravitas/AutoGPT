"""Tool for permanently remembering a fact for the current session's expert."""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .models import ErrorResponse, FactRememberedResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_PLAIN_SESSION_REFUSAL = (
    "I can only remember facts inside an expert's chat. Open a thread with a "
    "hired expert and I'll save it to their memory."
)


class RememberFactTool(BaseTool):
    """Append a durable fact to the current session's expert."""

    @property
    def name(self) -> str:
        return "remember_fact"

    @property
    def description(self) -> str:
        return (
            "Permanently remember a fact about the user, their business, or how "
            "this expert should work. Only for the expert whose chat this is. "
            "Saved facts are injected into every future session with this expert. "
            "Use for durable preferences and context, not one-off task details."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": (
                        "The fact to remember, as a short self-contained "
                        "sentence (e.g. 'The user prefers weekly reports on "
                        "Mondays')."
                    ),
                }
            },
            "required": ["fact"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Please sign in to save facts.",
                session_id=session_id,
            )
        if not session.expert_id:
            return ErrorResponse(message=_PLAIN_SESSION_REFUSAL, session_id=session_id)

        fact = str(kwargs.get("fact") or "").strip()
        if not fact:
            return ErrorResponse(
                message="Provide the fact to remember.",
                session_id=session_id,
            )

        expert = await experts_db().append_learned_note(
            user_id, session.expert_id, fact, source="chat"
        )
        newest = expert.learned_notes[-1]
        return FactRememberedResponse(
            message=f"Remembered: {fact}",
            session_id=session_id,
            note_id=newest.id,
            fact=fact,
            total_notes=len(expert.learned_notes),
        )
