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

        # expert and status filter in the query, so the scan limit bounds the
        # matches rather than the rows looked at; ``task`` reads ``purpose``
        # out of the metadata JSON and stays here.
        rows = await list_recent_chat_sessions(
            user_id=user_id,
            expert_id=expert_id.strip() or None,
            status=status.strip() or None,
            limit=_SCAN_LIMIT,
        )
        matched = [
            row
            for row in rows
            # The query already scopes to the caller; re-checking here keeps
            # the invariant with the tool rather than with one call site.
            if row.user_id == user_id
            and row.session_id != session.session_id
            and _matches_task(row, task.strip())
        ]
        shown = matched[:MAX_RESULTS]
        return SessionListResponse(
            message=_summary(len(shown), truncated=len(matched) > MAX_RESULTS),
            sessions=[
                SessionSummary(
                    session_id=row.session_id,
                    expert_id=row.expert_id,
                    title=row.title,
                    purpose=row.metadata.purpose,
                    status=row.chat_status,
                    updated_at=row.updated_at,
                )
                for row in shown
            ],
        )


def _matches_task(row: ChatSessionInfo, task: str) -> bool:
    if not task:
        return True
    haystack = f"{row.metadata.purpose or ''} {row.title or ''}".lower()
    return task.lower() in haystack


def _summary(shown: int, *, truncated: bool) -> str:
    """Count what was returned, not what matched: a number larger than the
    list reads as authoritative and is not."""
    if not shown:
        return "No other sessions of yours match."
    plural = "s" if shown != 1 else ""
    more = " Narrow it with expert_id, task or status." if truncated else ""
    return f"{shown} session{plural} of yours{' (first ' + str(MAX_RESULTS) + ')' if truncated else ''}.{more}"
