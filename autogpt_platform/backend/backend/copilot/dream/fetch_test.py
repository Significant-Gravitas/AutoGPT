from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatMessage

from . import fetch as fetch_mod


@pytest.mark.asyncio
async def test_autopilot_dream_fetches_only_unscoped_sessions(mocker):
    database = SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[]))
    mocker.patch.object(fetch_mod, "chat_db", return_value=database)

    await fetch_mod._fetch_recent_sessions(
        MemoryScope.for_user("user-1"), datetime.now(timezone.utc), 10
    )

    database.get_user_chat_sessions.assert_awaited_once_with(
        "user-1", limit=10, autopilot_only=True
    )


@pytest.mark.asyncio
async def test_expert_dream_fetches_only_that_experts_sessions(mocker):
    database = SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[]))
    mocker.patch.object(fetch_mod, "chat_db", return_value=database)

    await fetch_mod._fetch_recent_sessions(
        MemoryScope.for_expert("user-1", "expert-1"), datetime.now(timezone.utc), 10
    )

    database.get_user_chat_sessions.assert_awaited_once_with(
        "user-1", limit=10, expert_id="expert-1"
    )


@pytest.mark.asyncio
async def test_a_note_for_the_user_is_not_dreamt_about(mocker):
    """A skipped-run card is the platform talking to the user; a dream would
    store "usage limit reached" as a fact about them."""
    session = SimpleNamespace(session_id="session-1", title="Briefing")
    page = SimpleNamespace(
        messages=[
            ChatMessage(role="user", content="brief me"),
            ChatMessage(
                role="assistant",
                content="Skipped: daily usage limit reached",
                metadata={"hidden_from_model": True},
            ),
            ChatMessage(role="assistant", content="here is your brief"),
        ]
    )
    database = SimpleNamespace(
        get_user_chat_sessions=AsyncMock(return_value=[session]),
        get_chat_messages_paginated=AsyncMock(return_value=page),
    )
    mocker.patch.object(fetch_mod, "chat_db", return_value=database)

    rows = await fetch_mod._fetch_recent_sessions(
        MemoryScope.for_user("user-1"), datetime.now(timezone.utc), 10
    )

    assert rows[0].body == "user: brief me\nassistant: here is your brief"
