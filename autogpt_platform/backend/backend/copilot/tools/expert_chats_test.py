"""Tests for list_expert_chats / read_expert_chat (Autopilot reads an expert's chats).

The capability is one-directional and read-only, so what is tested here is
mostly what the tools refuse: an expert session never sees them and is turned
away if it names one anyway, another user's chat is indistinguishable from one
that does not exist, and a chat outside the listing's scope (Autopilot's own,
a dream artifact) cannot be reached by id. The rest pins the paging contract —
the character cap drops rows from the OLD end so the cursor it reports brings
them back.

``_FakeChatDB`` reproduces the one behaviour the ownership check rests on:
``get_chat_messages_paginated`` puts ``user_id`` in the ChatSession
where-clause, so a foreign caller gets ``None``. A mock that ignored it would
make the ownership tests pass with the check deleted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage, ChatSessionInfo, ChatSessionMetadata
from backend.copilot.tools import (
    TOOL_GROUPS,
    execute_tool,
    get_available_tools,
    get_tool,
)

from .expert_chats import _MAX_PAGE_CHARS, ListExpertChatsTool, ReadExpertChatTool
from .models import ErrorResponse, ExpertChatListResponse, ExpertChatTranscriptResponse

_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def _caller(session_id: str = "s1", expert_id: str | None = None) -> MagicMock:
    """The Autopilot session the tool is called from."""
    session = MagicMock()
    session.session_id = session_id
    session.expert_id = expert_id
    session.organization_id = None
    return session


def _chat(
    session_id: str = "chat-1",
    *,
    user_id: str = "alice",
    expert_id: str | None = "expert-a",
    title: str | None = "Q3 research",
    kind: str = "normal",
) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id=user_id,
        title=title,
        usage=[],
        started_at=_NOW,
        updated_at=_NOW,
        metadata=ChatSessionMetadata(kind=kind),
        expert_id=expert_id,
    )


def _msg(
    sequence: int,
    role: str = "assistant",
    content: str | None = "hello",
    **kwargs,
) -> ChatMessage:
    return ChatMessage(
        role=role, content=content, sequence=sequence, created_at=_NOW, **kwargs
    )


class _FakeChatDB:
    """The two query behaviours the tools depend on, ownership included."""

    def __init__(
        self,
        chats: list[ChatSessionInfo] | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> None:
        self.chats = chats if chats is not None else [_chat()]
        self.messages = messages or []
        self.list_calls: list[dict] = []

    async def get_user_chat_sessions(self, user_id, limit, offset, **kwargs):
        self.list_calls.append({"user_id": user_id, "limit": limit, **kwargs})
        rows = [c for c in self.chats if c.user_id == user_id]
        if expert_id := kwargs.get("expert_id"):
            rows = [c for c in rows if c.expert_id == expert_id]
        elif kwargs.get("experts_only"):
            rows = [c for c in rows if c.expert_id is not None]
        return rows[offset : offset + limit]

    async def get_chat_messages_paginated(
        self,
        session_id,
        limit=50,
        before_sequence=None,
        user_id=None,
        organization_id=None,
    ):
        chat = next((c for c in self.chats if c.session_id == session_id), None)
        # Mirrors db.py: ``user_id``, when given, is part of the session
        # where-clause — a chat owned by anyone else is simply not found.
        if chat is None or (user_id is not None and chat.user_id != user_id):
            return None
        rows = [
            m
            for m in self.messages
            if before_sequence is None or (m.sequence or 0) < before_sequence
        ]
        window = rows[-limit:] if limit else rows
        return PaginatedMessages(
            messages=window,
            has_more=len(window) < len(rows),
            oldest_sequence=window[0].sequence if window else None,
            session=chat,
        )


def _experts(*pairs: tuple[str, str]) -> MagicMock:
    roster = []
    for expert_id, name in pairs:
        expert = MagicMock()
        expert.id = expert_id
        expert.name = name
        roster.append(expert)
    client = MagicMock()
    client.list_experts = AsyncMock(return_value=roster)
    client.get_expert = AsyncMock(return_value=roster[0] if roster else None)
    return MagicMock(return_value=client)


def _patches(db: _FakeChatDB, *pairs: tuple[str, str]):
    return (
        patch("backend.copilot.tools.expert_chats.chat_db", MagicMock(return_value=db)),
        patch(
            "backend.copilot.tools.expert_chats.experts_db",
            _experts(*(pairs or (("expert-a", "Ada"),))),
        ),
    )


async def _read(db: _FakeChatDB, **kwargs):
    chat_patch, experts_patch = _patches(db)
    with chat_patch, experts_patch:
        return await ReadExpertChatTool()._execute(
            "alice", _caller(), session_id="chat-1", **kwargs
        )


async def _list(db: _FakeChatDB, **kwargs):
    chat_patch, experts_patch = _patches(db)
    with chat_patch, experts_patch:
        return await ListExpertChatsTool()._execute("alice", _caller(), **kwargs)


class TestGating:
    """Autopilot-only: the tools ride the ``expert_admin`` group, which the
    engines disable for every expert session."""

    def test_both_tools_are_in_the_autopilot_only_group(self) -> None:
        assert TOOL_GROUPS["list_expert_chats"] == "expert_admin"
        assert TOOL_GROUPS["read_expert_chat"] == "expert_admin"

    def test_an_autopilot_session_is_offered_them_and_an_expert_session_is_not(
        self,
    ) -> None:
        autopilot = {t["function"]["name"] for t in get_available_tools()}
        # What survives the filter, NOT what it hid — naming this `hidden`
        # invites "fixing" the assertion below into its own inverse.
        expert = {
            t["function"]["name"]
            for t in get_available_tools(disabled_groups=["expert_admin"])
        }
        assert {"list_expert_chats", "read_expert_chat"} <= autopilot
        assert not {"list_expert_chats", "read_expert_chat"} & expert

    @pytest.mark.parametrize("tool_name", ["list_expert_chats", "read_expert_chat"])
    @pytest.mark.asyncio
    async def test_an_expert_session_naming_the_tool_is_refused_before_dispatch(
        self, tool_name: str
    ) -> None:
        tool = get_tool(tool_name)
        assert tool is not None
        with patch.object(
            tool, "execute", new=AsyncMock(return_value="should never run")
        ) as execute_mock:
            result = await execute_tool(
                tool_name=tool_name,
                parameters={"session_id": "chat-1"},
                user_id="alice",
                session=_caller(expert_id="expert-a"),
                tool_call_id="call-1",
                disabled_groups=["expert_admin"],
            )
        execute_mock.assert_not_awaited()
        assert result.success is False
        assert ErrorResponse.model_validate_json(result.output).error == "tool_disabled"


class TestListExpertChats:
    @pytest.mark.asyncio
    async def test_listing_without_an_expert_id_asks_for_expert_chats_only(
        self,
    ) -> None:
        db = _FakeChatDB(chats=[_chat(), _chat("chat-2", expert_id=None)])
        result = await _list(db)
        assert isinstance(result, ExpertChatListResponse)
        assert db.list_calls[0]["experts_only"] is True
        assert [c.session_id for c in result.chats] == ["chat-1"]

    @pytest.mark.asyncio
    async def test_an_expert_id_filters_and_never_pairs_with_experts_only(
        self,
    ) -> None:
        db = _FakeChatDB(chats=[_chat(), _chat("chat-2", expert_id="expert-b")])
        result = await _list(db, expert_id="expert-b")
        assert db.list_calls[0]["expert_id"] == "expert-b"
        # The two are mutually exclusive in the query; passing both raises.
        assert db.list_calls[0]["experts_only"] is False
        assert [c.session_id for c in result.chats] == ["chat-2"]

    @pytest.mark.asyncio
    async def test_rows_carry_the_expert_name_not_just_the_id(self) -> None:
        result = await _list(_FakeChatDB())
        assert result.chats[0].expert_name == "Ada"
        assert "Ada" in result.message

    @pytest.mark.asyncio
    async def test_another_users_chats_are_never_listed(self) -> None:
        db = _FakeChatDB(chats=[_chat("chat-9", user_id="mallory")])
        result = await _list(db)
        assert result.chats == []


class TestReadOwnership:
    @pytest.mark.asyncio
    async def test_another_users_chat_reads_exactly_like_a_missing_one(self) -> None:
        foreign = await _read(_FakeChatDB(chats=[_chat(user_id="mallory")]))
        missing = await _read(_FakeChatDB(chats=[]))
        assert isinstance(foreign, ErrorResponse)
        assert foreign.message == missing.message

    @pytest.mark.asyncio
    async def test_the_callers_user_id_reaches_the_query(self) -> None:
        """The ownership check IS that argument — pin it, not just its effect."""
        db = _FakeChatDB(messages=[_msg(1)])
        seen: dict = {}
        original = db.get_chat_messages_paginated

        async def spy(*args, **kwargs):
            seen.update(kwargs)
            return await original(*args, **kwargs)

        db.get_chat_messages_paginated = spy  # type: ignore[method-assign]
        await _read(db)
        assert seen["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_an_autopilot_chat_cannot_be_reached_by_id(self) -> None:
        result = await _read(_FakeChatDB(chats=[_chat(expert_id=None)]))
        assert isinstance(result, ErrorResponse)

    @pytest.mark.asyncio
    async def test_a_dream_session_cannot_be_reached_by_id(self) -> None:
        result = await _read(_FakeChatDB(chats=[_chat(kind="dream")]))
        assert isinstance(result, ErrorResponse)


class TestReadRendering:
    @pytest.mark.asyncio
    async def test_it_shows_what_the_user_sees_and_hides_what_they_do_not(
        self,
    ) -> None:
        db = _FakeChatDB(
            messages=[
                # The trailing blank line is part of the injected shape the
                # production regex anchors on, not test decoration.
                _msg(1, "user", "<user_context>secret</user_context>\n\nreal question"),
                _msg(2, "user", "kickoff", metadata={"hidden": True}),
                _msg(3, "tool", '{"bulk": "output"}', tool_call_id="c1"),
                _msg(
                    4,
                    "assistant",
                    "on it",
                    tool_calls=[{"function": {"name": "run_agent"}}],
                ),
            ]
        )
        result = await _read(db)
        assert isinstance(result, ExpertChatTranscriptResponse)
        assert [m.sequence for m in result.messages] == [1, 4]
        assert result.messages[0].content == "real question"
        assert "[called: run_agent]" in result.messages[1].content

    @pytest.mark.asyncio
    async def test_tool_output_is_available_on_request(self) -> None:
        db = _FakeChatDB(messages=[_msg(1, "tool", "bulk", tool_call_id="c1")])
        result = await _read(db, include_tool_results=True)
        assert [m.sequence for m in result.messages] == [1]


class TestReadPaging:
    def _long_chat(self) -> _FakeChatDB:
        return _FakeChatDB(messages=[_msg(i, content="x" * 2_500) for i in range(1, 6)])

    @pytest.mark.asyncio
    async def test_the_cap_drops_the_oldest_rows_and_reports_them_as_more(
        self,
    ) -> None:
        result = await _read(self._long_chat())
        assert sum(len(m.content) for m in result.messages) <= _MAX_PAGE_CHARS
        assert [m.sequence for m in result.messages] == [3, 4, 5]
        assert result.has_more is True
        assert result.next_before_sequence == 3

    @pytest.mark.asyncio
    async def test_the_reported_cursor_brings_the_dropped_rows_back(self) -> None:
        db = self._long_chat()
        first = await _read(db)
        older = await _read(db, before_sequence=first.next_before_sequence)
        assert [m.sequence for m in older.messages] == [1, 2]
        assert older.has_more is False

    @pytest.mark.asyncio
    async def test_a_short_chat_says_so_instead_of_offering_a_cursor(self) -> None:
        result = await _read(_FakeChatDB(messages=[_msg(1), _msg(2)]))
        assert result.has_more is False
        assert "start of the chat" in result.message
