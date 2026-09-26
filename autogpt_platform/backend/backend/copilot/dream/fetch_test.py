from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.recall import (
    forgotten_facts_clause,
    recallable_episode_predicate,
)
from backend.copilot.graphiti.scope import MemoryScope

from . import fetch as fetch_mod
from . import hidden_sessions


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
async def test_episode_gather_reads_only_recallable_episodes():
    driver = AsyncMock()
    driver.execute_query.return_value = ([], [], None)

    await fetch_mod._fetch_recent_episodes(
        driver, "user_g", datetime.now(timezone.utc), 50
    )

    query = driver.execute_query.await_args.args[0]
    assert query.startswith(forgotten_facts_clause())
    assert f"WHERE {recallable_episode_predicate('n')}" in query


def _chat_store(*session_ids: str) -> SimpleNamespace:
    message = SimpleNamespace(role="user", content="something the user said")
    return SimpleNamespace(
        get_user_chat_sessions=AsyncMock(
            return_value=[
                SimpleNamespace(session_id=sid, title=sid) for sid in session_ids
            ]
        ),
        get_chat_messages_paginated=AsyncMock(
            return_value=SimpleNamespace(messages=[message])
        ),
    )


def _graph(hidden_rows: list[dict] | Exception) -> AsyncMock:
    """A driver whose hidden-episode read returns (or raises) ``hidden_rows``;
    the episode and fact reads find nothing."""

    async def execute(query: str, **params: object):
        if query != hidden_sessions._HIDDEN_EPISODES_QUERY:
            return ([], [], None)
        if isinstance(hidden_rows, Exception):
            raise hidden_rows
        return (hidden_rows, [], None)

    driver = AsyncMock()
    driver.execute_query.side_effect = execute
    return driver


@pytest.mark.asyncio
async def test_the_dream_reads_no_session_a_forget_hid(mocker):
    forgotten = {
        "name": "conversation_s-gone",
        "source_description": "",
        "content": "",
        "provenance": None,
    }
    mocker.patch.object(fetch_mod, "open_driver", return_value=_graph([forgotten]))
    chat_store = _chat_store("s-gone", "s-kept")
    mocker.patch.object(fetch_mod, "chat_db", return_value=chat_store)

    bundle = await fetch_mod.gather_dream_input(MemoryScope.for_user("user-1"))

    assert [session.session_id for session in bundle.recent_sessions] == ["s-kept"]
    chat_store.get_chat_messages_paginated.assert_awaited_once_with(
        session_id="s-kept", limit=20, user_id="user-1"
    )


@pytest.mark.asyncio
async def test_the_dream_reads_no_session_when_it_cannot_tell_which_are_hidden(
    mocker,
):
    graph = _graph(RuntimeError("falkordb down"))
    mocker.patch.object(fetch_mod, "open_driver", return_value=graph)
    chat_store = _chat_store("s-1")
    mocker.patch.object(fetch_mod, "chat_db", return_value=chat_store)

    bundle = await fetch_mod.gather_dream_input(MemoryScope.for_user("user-1"))

    assert bundle.recent_sessions == []
    chat_store.get_user_chat_sessions.assert_not_awaited()
