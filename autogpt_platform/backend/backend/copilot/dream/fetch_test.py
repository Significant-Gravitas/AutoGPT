from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.recall import (
    forgotten_facts_clause,
    live_fact_predicate,
    recallable_episode_predicate,
)
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.legacy_first_turn_memory_test_data import (
    legacy_first_message,
    master_warm,
)
from backend.copilot.model import ChatMessage

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


@pytest.mark.asyncio
async def test_fact_gather_reads_live_facts_by_recalls_own_test():
    """A forgotten fact (``forgotten_at`` set) never reaches the dream, even
    one some writer left unexpired."""
    driver = AsyncMock()
    driver.execute_query.return_value = ([], [], None)

    await fetch_mod._fetch_active_facts(driver, "user_g", 50)

    query = driver.execute_query.await_args.args[0]
    assert f"WHERE {live_fact_predicate('e', include_tentative=False)}" in query


def _chat_store(*session_ids: str) -> SimpleNamespace:
    message = ChatMessage(role="user", content="something the user said", sequence=0)
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


def test_a_session_body_reads_the_first_message_without_its_stored_block():
    """The dream's session bodies are model input: the first message comes
    without the memory block an older session stored in it, the backfill not
    yet run; a later message holding a copy (a paste) is the user's."""
    first = legacy_first_message(master_warm(("the Nova password is violet-913",)))
    pasted = legacy_first_message(master_warm(("Bob leads Atlas",)))

    body = fetch_mod._session_body(
        [
            ChatMessage(role="user", content=first, sequence=0),
            ChatMessage(role="assistant", content="done", sequence=1),
            ChatMessage(role="user", content=pasted, sequence=2),
        ]
    )

    assert "violet-913" not in body
    assert "what is Alice working on" in body
    assert "Bob leads Atlas" in body
