"""Read the user's chats with their hired experts — AutoPilot only, never writes.

An expert's chat is the user's own data, but it lives outside the AutoPilot
thread and nothing in the session context carries it. These two tools give
AutoPilot the read the user already has in the UI: the same paginated query,
so the same ownership check, and the same hidden-row and injected-context
filtering the chat API applies before rendering.
"""

import logging
from typing import Any

from backend.copilot.expert_kickoff import is_hidden_chat_message
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.service import strip_injected_context_for_display
from backend.data.db_accessors import chat_db, experts_db
from backend.util.truncate import truncate

from .base import BaseTool
from .models import (
    ErrorResponse,
    ExpertChatListResponse,
    ExpertChatMessage,
    ExpertChatSummary,
    ExpertChatTranscriptResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

_MAX_LIMIT = 50
_DEFAULT_LIMIT = 20

# Two budgets, because a single transcript row can be a 95K tool result: the
# page cap bounds what one call costs the turn, the row cap stops one row
# from spending the whole page. Keep the row cap well under the page cap so
# a page always holds several messages.
_MAX_PAGE_CHARS = 8_000
_MAX_MESSAGE_CHARS = 2_000

# Same wording for "no such chat" and "belongs to someone else", so the tool
# is not an existence oracle for session ids.
_NOT_FOUND = (
    "No expert chat with id {session_id}. Call list_expert_chats for the ids "
    "you can read."
)


class ListExpertChatsTool(BaseTool):
    """List the user's chats with their hired experts."""

    @property
    def name(self) -> str:
        return "list_expert_chats"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "List the user's chats with their hired experts, most recently "
            "updated first. Pass expert_id to see one expert's chats only. "
            "Returns session ids for read_expert_chat; never returns message "
            "content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": (
                        "Only chats with this expert. Omit for every expert. "
                        "Ids come from list_team or <team_context>."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": f"Chats to return (max {_MAX_LIMIT}).",
                    "default": _DEFAULT_LIMIT,
                },
                "offset": {
                    "type": "integer",
                    "description": "Chats to skip, for paging.",
                    "default": 0,
                },
            },
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str | None = None,
        limit: int = _DEFAULT_LIMIT,
        offset: int = 0,
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return ErrorResponse(
                message="Authentication required", session_id=session.session_id
            )

        one_expert = (expert_id or "").strip() or None
        page_size = _clamp(limit, _DEFAULT_LIMIT)
        start = max(0, offset or 0)
        try:
            # One row past the page, so a full page can say whether it is the
            # last one rather than costing a call to find out.
            chats = await chat_db().get_user_chat_sessions(
                user_id,
                page_size + 1,
                start,
                organization_id=session.organization_id,
                expert_id=one_expert,
                # Mutually exclusive with expert_id, which already implies it.
                experts_only=one_expert is None,
            )
        except Exception as e:
            logger.warning(f"list_expert_chats lookup failed: {e}")
            return ErrorResponse(
                message="Could not load the expert chats right now. Try again.",
                session_id=session.session_id,
            )

        has_more = len(chats) > page_size
        chats = chats[:page_size]

        names = await _expert_names(user_id)
        if not chats:
            who = names.get(one_expert, one_expert) if one_expert else "any expert"
            return ExpertChatListResponse(
                message=f"No chats with {who} yet.",
                session_id=session.session_id,
            )

        rows = [
            ExpertChatSummary(
                session_id=chat.session_id,
                # experts_only / expert_id guarantee a non-null expertId.
                expert_id=chat.expert_id or "",
                expert_name=names.get(chat.expert_id or ""),
                title=chat.title,
                updated_at=chat.updated_at,
            )
            for chat in chats
        ]
        listing = "; ".join(
            f"{row.expert_name or row.expert_id} — "
            f"{row.title or 'untitled'} "
            f"(session_id: {row.session_id}, "
            f"updated {row.updated_at:%Y-%m-%d})"
            for row in rows
        )
        next_offset = start + len(rows) if has_more else None
        more = (
            f" More chats follow — pass offset {next_offset} for the next page."
            if next_offset is not None
            else ""
        )
        return ExpertChatListResponse(
            message=(
                f"{len(rows)} expert chat{'s' if len(rows) != 1 else ''}: "
                f"{listing}. Read one with read_expert_chat.{more}"
            ),
            session_id=session.session_id,
            chats=rows,
            has_more=has_more,
            next_offset=next_offset,
        )


class ReadExpertChatTool(BaseTool):
    """Read a window of one expert chat."""

    @property
    def name(self) -> str:
        return "read_expert_chat"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Read one of the user's chats with a hired expert, newest "
            "messages first. Ids come from list_expert_chats. Walk back "
            "through a long chat by passing the returned "
            "next_before_sequence; ask for the window you need rather than "
            "the whole chat."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Chat id from list_expert_chats.",
                },
                "before_sequence": {
                    "type": "integer",
                    "description": (
                        "Return messages older than this sequence. Omit for "
                        "the newest ones; then pass back the response's "
                        "next_before_sequence to page further back."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Messages to fetch (max {_MAX_LIMIT}). Fewer are "
                        f"returned when they exceed the {_MAX_PAGE_CHARS:,}"
                        "-character budget."
                    ),
                    "default": _DEFAULT_LIMIT,
                },
                "include_tool_results": {
                    "type": "boolean",
                    "description": (
                        "Include the expert's raw tool output. Off by "
                        "default: tool calls are already named on the "
                        "assistant message, and their results are bulky."
                    ),
                    "default": False,
                },
            },
            "required": ["session_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        session_id: str = "",
        before_sequence: int | None = None,
        limit: int = _DEFAULT_LIMIT,
        include_tool_results: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return ErrorResponse(
                message="Authentication required", session_id=session.session_id
            )
        chat_id = (session_id or "").strip()
        if not chat_id:
            return ErrorResponse(
                message="session_id is required",
                session_id=session.session_id,
            )

        try:
            page = await chat_db().get_chat_messages_paginated(
                chat_id,
                _clamp(limit, _DEFAULT_LIMIT),
                before_sequence,
                # The ownership check: a chat owned by anyone else, or one
                # outside the caller's org scope, comes back as None.
                user_id=user_id,
                organization_id=session.organization_id,
            )
        except Exception as e:
            logger.warning(f"read_expert_chat lookup failed for {chat_id}: {e}")
            return ErrorResponse(
                message="Could not load that chat right now. Try again.",
                session_id=session.session_id,
            )

        # Both are excluded from list_expert_chats, so reject them here too —
        # otherwise fetch-by-id is a way around the listing's scope.
        if (
            page is None
            or page.session.expert_id is None
            or page.session.metadata.kind == "dream"
        ):
            return ErrorResponse(
                message=_NOT_FOUND.format(session_id=chat_id),
                session_id=session.session_id,
            )

        rendered = [
            message
            for message in (
                _render(m, include_tool_results=include_tool_results)
                for m in page.messages
            )
            if message is not None
        ]
        kept, over_budget = _fit_budget(rendered)
        has_more = page.has_more or over_budget
        next_cursor = kept[0].sequence if kept else page.oldest_sequence

        expert_name = await _expert_name(user_id, page.session.expert_id)
        title = page.session.title or "untitled"
        if not kept:
            summary = f"No readable messages in {expert_name}'s chat “{title}”."
        else:
            summary = (
                f"{len(kept)} message{'s' if len(kept) != 1 else ''} from "
                f"{expert_name}'s chat “{title}”, oldest first "
                f"(sequences {kept[0].sequence}–{kept[-1].sequence})."
            )
        if has_more and next_cursor is not None:
            summary += (
                " Older messages remain — call read_expert_chat again with "
                f"before_sequence={next_cursor}."
            )
        elif kept:
            summary += " This is the start of the chat."

        return ExpertChatTranscriptResponse(
            message=summary,
            session_id=session.session_id,
            chat_session_id=chat_id,
            expert_id=page.session.expert_id,
            expert_name=expert_name,
            title=page.session.title,
            messages=kept,
            has_more=has_more,
            next_before_sequence=next_cursor,
        )


def _clamp(value: int | None, default: int) -> int:
    return max(1, min(value if value is not None else default, _MAX_LIMIT))


def _render(
    message: ChatMessage, *, include_tool_results: bool
) -> ExpertChatMessage | None:
    """Project one persisted row into what the user sees, or None to skip it.

    Mirrors the chat API's own read path: control turns the UI hides stay
    hidden, and the server-injected ``<…_context>`` prefixes are stripped
    from user rows.
    """
    if message.sequence is None or is_hidden_chat_message(message.metadata):
        return None
    if message.role == "tool" and not include_tool_results:
        return None
    if message.role not in ("user", "assistant", "tool"):
        return None

    content = message.content or ""
    if message.role == "user":
        content = strip_injected_context_for_display(content)
    if message.role == "assistant" and message.tool_calls:
        names = ", ".join(
            (call.get("function") or {}).get("name") or call.get("name") or "?"
            for call in message.tool_calls
        )
        content = f"{content}\n[called: {names}]".strip()
    if not content.strip():
        return None

    return ExpertChatMessage(
        sequence=message.sequence,
        role=message.role,
        content=truncate(content, _MAX_MESSAGE_CHARS),
        created_at=message.created_at,
    )


def _fit_budget(
    messages: list[ExpertChatMessage],
) -> tuple[list[ExpertChatMessage], bool]:
    """Keep the newest messages that fit ``_MAX_PAGE_CHARS``, oldest dropped first.

    Dropping from the old end is what keeps paging lossless: the caller's
    next cursor is the oldest row it actually received, so the dropped rows
    are the first thing the next page returns.
    """
    kept: list[ExpertChatMessage] = []
    spent = 0
    for message in reversed(messages):
        spent += len(message.content)
        if spent > _MAX_PAGE_CHARS and kept:
            return kept, True
        kept.insert(0, message)
    return kept, False


async def _expert_names(user_id: str) -> dict[str, str]:
    """Expert id → name, so a listing reads as a person rather than a uuid."""
    try:
        experts = await experts_db().list_experts(user_id, with_metrics=False)
    except Exception as e:
        logger.warning(f"expert roster lookup failed: {e}")
        return {}
    return {expert.id: expert.name for expert in experts}


async def _expert_name(user_id: str, expert_id: str) -> str:
    try:
        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False
        )
    except Exception as e:
        logger.warning(f"expert lookup failed for {expert_id}: {e}")
        return expert_id
    return expert.name if expert else expert_id
