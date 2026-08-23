"""Next-step chips offered to the user after a substantive turn.

The model supplies up to three short imperative labels; this tool
normalises them and republishes them onto the live turn stream as a
``data-suggestions`` part, so the client renders chips from a typed
``message.parts`` entry instead of parsing tool output.

Emitting from the tool (rather than from ``mark_session_completed``) keeps
the chips authored by the model — the same way ``TodoWrite`` lets the model
own the checklist — and reuses the mid-turn publish path already used by
``pending_messages._notify_pending_drained``.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    MAX_SUGGESTION_LENGTH,
    MAX_SUGGESTIONS,
    StreamSuggestions,
)
from backend.copilot.stream_registry import get_session, publish_chunk

from .base import BaseTool
from .models import ErrorResponse, SuggestNextStepsResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class SuggestNextStepsTool(BaseTool):
    """Offer the user up to three one-tap follow-up actions."""

    @property
    def name(self) -> str:
        return "suggest_next_steps"

    @property
    def description(self) -> str:
        return (
            "Offer the user up to 3 one-tap next actions as chips, right "
            "before your closing summary. Only after substantive work, and "
            "only for actions you can carry out yourself if tapped."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "description": (
                        "Short imperative labels, e.g. 'Email the report'. "
                        f"At most {MAX_SUGGESTIONS}; extras are dropped."
                    ),
                    "maxItems": MAX_SUGGESTIONS,
                    "items": {
                        "type": "string",
                        "maxLength": MAX_SUGGESTION_LENGTH,
                    },
                },
            },
            "required": ["suggestions"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id
        raw = kwargs.get("suggestions")
        if not isinstance(raw, list):
            return ErrorResponse(
                message="`suggestions` must be an array of strings.",
                session_id=session.session_id,
            )
        if any(not isinstance(item, str) for item in raw):
            return ErrorResponse(
                message="`suggestions` entries must be strings.",
                session_id=session.session_id,
            )

        event = StreamSuggestions(suggestions=raw)
        if not event.suggestions:
            return ErrorResponse(
                message="`suggestions` contained no usable labels.",
                session_id=session.session_id,
            )

        await _publish_suggestions(session.session_id, event)
        return SuggestNextStepsResponse(
            message="Next-step chips shown to the user.",
            session_id=session.session_id,
            suggestions=event.suggestions,
        )


async def _publish_suggestions(
    session_id: str | None, event: StreamSuggestions
) -> None:
    """Push the chips onto the session's live turn stream.

    Best-effort: chips are a convenience surface, so a failed publish must
    never fail the turn the model just completed.
    """
    if not session_id:
        return
    try:
        active = await get_session(session_id)
        if active is None or not active.turn_id:
            return
        await publish_chunk(active.turn_id, event, session_id=session_id)
    except Exception:
        logger.debug(
            "suggest_next_steps: chip emit failed for session=%s",
            session_id,
            exc_info=True,
        )
