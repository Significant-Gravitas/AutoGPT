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
# ``task`` reads the metadata JSON, so it is the one filter that stays in
# Python — which means the scan has to keep paging until it has MAX_RESULTS
# matches, or a match older than the first page is reported as no match at all.
_MAX_SCAN_PAGES = 5


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
            "running. Use with tool:message_session."
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
                    "description": "Match what the session is for, or its title. Searches your recent sessions only.",
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
        matched: list[ChatSessionInfo] = []
        scanned = 0
        exhausted = False
        for _ in range(_MAX_SCAN_PAGES):
            rows = await list_recent_chat_sessions(
                user_id=user_id,
                expert_id=expert_id.strip() or None,
                status=status.strip() or None,
                limit=_SCAN_LIMIT,
                skip=scanned,
            )
            scanned += len(rows)
            matched.extend(
                row
                for row in rows
                # The query already scopes to the caller; re-checking here
                # keeps the invariant with the tool rather than with one call
                # site.
                if row.user_id == user_id
                and row.session_id != session.session_id
                and _matches_task(row, task.strip())
            )
            if len(rows) < _SCAN_LIMIT:
                exhausted = True
                break
            if len(matched) > MAX_RESULTS:
                break
        shown = matched[:MAX_RESULTS]
        return SessionListResponse(
            message=_summary(
                len(shown),
                truncated=len(matched) > MAX_RESULTS,
                window_full=not exhausted,
            ),
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


def _summary(shown: int, *, truncated: bool, window_full: bool) -> str:
    """Count what was returned, not what matched: a number larger than the
    list reads as authoritative and is not.

    Nothing found is the same trap one step further on. ``task`` is matched
    after the scan, so an empty result that stopped at the page cap means "not
    among the recent ones", which is not the same answer as "you have none".
    """
    if not shown:
        if window_full:
            return "No match among your recent sessions. Try expert_id or status."
        return "No other sessions of yours match."
    plural = "s" if shown != 1 else ""
    more = " Narrow it with expert_id, task or status." if truncated else ""
    return f"{shown} session{plural} of yours{' (first ' + str(MAX_RESULTS) + ')' if truncated else ''}.{more}"
