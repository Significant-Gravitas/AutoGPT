"""heartbeat_respond: how a heartbeat turn says whether to alert the user.

Only a heartbeat session (``ChatSessionMetadata.kind == "heartbeat"``) may
call it. The tool records the answer; ``heartbeat.runner`` reads it after the
turn and decides delivery, suppression and dedupe, so nothing reaches the user
from inside the turn itself.
"""

import logging
from typing import Any

from backend.copilot.heartbeat.state import HEARTBEAT_SESSION_KIND, record_response
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, HeartbeatRespondResponse, ToolResponseBase

logger = logging.getLogger(__name__)

MAX_NOTIFICATION_CHARS = 1_000


class HeartbeatRespondTool(BaseTool):
    @property
    def name(self) -> str:
        return "heartbeat_respond"

    @property
    def description(self) -> str:
        return "Heartbeat runs only: notify=true alerts the user; false is quiet."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "notify": {"type": "boolean", "description": "Alert the user."},
                "notification_text": {"type": "string", "description": "The alert."},
            },
            "required": ["notify"],
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
        session_id = session.session_id if session else None
        if session is None or session.metadata.kind != HEARTBEAT_SESSION_KIND:
            return ErrorResponse(
                message="heartbeat_respond only works in a heartbeat run.",
                error="not_a_heartbeat",
                session_id=session_id,
            )
        notify = bool(kwargs.get("notify"))
        text = str(kwargs.get("notification_text") or "").strip()
        if notify and not text:
            return ErrorResponse(
                message="notification_text is required when notify is true.",
                error="missing_notification_text",
                session_id=session_id,
            )
        text = text[:MAX_NOTIFICATION_CHARS]
        if not await record_response(session.session_id, notify, text):
            return ErrorResponse(
                message="The answer could not be recorded; reply NO_REPLY.",
                error="not_recorded",
                session_id=session_id,
            )
        return HeartbeatRespondResponse(
            message=(
                "Recorded; the user will be alerted. End the turn now."
                if notify
                else "Recorded; nothing will be sent. Reply NO_REPLY."
            ),
            notify=notify,
            session_id=session_id,
        )
