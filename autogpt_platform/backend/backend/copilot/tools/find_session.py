"""Find one of the user's own sessions to message.

The registry is the ``ChatSession`` row itself — it already carries the stable
id, the owner, the bound expert and the live status — so this tool is a scoped
query, not a new store. ``user_id`` is applied in the query rather than as a
filter over a wider result, which is what makes another user's session
invisible instead of merely unlisted.
"""

import logging
from typing import Any

from backend.copilot.db import list_recent_chat_sessions
from backend.copilot.model import ChatSession, ChatSessionInfo

from .base import BaseTool
from .models import ErrorResponse, SessionListResponse, SessionSummary, ToolResponseBase

logger = logging.getLogger(__name__)

MAX_RESULTS = 20
_SCAN_LIMIT = 50


class FindSessionTool(BaseTool):
    """List the caller's own sessions, filtered by expert, purpose or status."""

    @property
    def name(self) -> str:
        return "find_session"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "List your own live sessions to find one to message: session id, "
            "the expert it is bound to, what it is for, and whether it is "
            "running. Use with message_session."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": "Only sessions bound to this expert.",
                    "default": "",
                },
                "task": {
                    "type": "string",
                    "description": "Match against what the session is for, and its title.",
                    "default": "",
                },
                "status": {
                    "type": "string",
                    "enum": ["idle", "queued", "running"],
                    "description": "Only sessions in this state.",
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str = "",
        task: str = "",
        status: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return ErrorResponse(
                message="Authentication required", session_id=session.session_id
            )

        rows = await list_recent_chat_sessions(user_id=user_id, limit=_SCAN_LIMIT)
        matched = [
            row
            for row in rows
            # The query already scopes to the caller; re-checking here keeps
            # the invariant with the tool rather than with one call site.
            if row.user_id == user_id
            and row.session_id != session.session_id
            and _matches(row, expert_id.strip(), task.strip(), status.strip())
        ]
        return SessionListResponse(
            message=_summary(len(matched)),
            sessions=[
                SessionSummary(
                    session_id=row.session_id,
                    expert_id=row.expert_id,
                    title=row.title,
                    purpose=row.metadata.purpose,
                    status=row.chat_status,
                    updated_at=row.updated_at,
                )
                for row in matched[:MAX_RESULTS]
            ],
        )


def _matches(row: ChatSessionInfo, expert_id: str, task: str, status: str) -> bool:
    if expert_id and row.expert_id != expert_id:
        return False
    if status and row.chat_status != status:
        return False
    if task:
        haystack = f"{row.metadata.purpose or ''} {row.title or ''}".lower()
        if task.lower() not in haystack:
            return False
    return True


def _summary(count: int) -> str:
    if not count:
        return "No other sessions of yours match."
    return f"{count} session{'s' if count != 1 else ''} of yours match."
