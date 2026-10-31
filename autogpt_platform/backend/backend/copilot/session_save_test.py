"""How a chat session's save writes the session row: where its rows carry the
session's own fields into their transaction, and where the fields keep an
update of their own."""

import pytest
from pytest_mock import MockerFixture

from backend.copilot import db as chat_db_module
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    ChatSessionSaveFields,
    Usage,
    _save_session_to_db,
    get_chat_session,
    invalidate_session_cache,
    upsert_chat_session,
)


def _make_session_with_messages(*msgs: ChatMessage) -> ChatSession:
    s = ChatSession.new(user_id="u1", dry_run=False)
    s.messages = list(msgs)
    return s


def _recording_chat_db(mocker: MockerFixture, *, local: bool) -> list[tuple[str, dict]]:
    """Patch the chat database with one that records each write, in order;
    ``local`` is whether it is this process's own Prisma client."""
    writes: list[tuple[str, dict]] = []

    async def _update_session(**kwargs) -> None:
        writes.append(("session", kwargs))

    async def _add_rows(**kwargs) -> int:
        writes.append(("rows", kwargs))
        return 1

    mock_db = mocker.MagicMock()
    mock_db.update_chat_session = mocker.AsyncMock(side_effect=_update_session)
    mock_db.add_chat_messages_batch = mocker.AsyncMock(side_effect=_add_rows)
    mocker.patch("backend.copilot.model.chat_db", return_value=mock_db)
    mocker.patch("backend.copilot.model.chat_db_is_local", return_value=local)
    return writes


def _session_with_a_new_reply() -> ChatSession:
    return _make_session_with_messages(
        ChatMessage(role="user", content="hi", sequence=0),
        ChatMessage(role="assistant", content="hello"),
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_save_with_new_messages_writes_the_session_fields_with_them(
    mocker: MockerFixture,
) -> None:
    """One write per message save: the session's fields ride in the rows'
    transaction rather than in an update of their own ahead of it."""
    writes = _recording_chat_db(mocker, local=True)
    session = _session_with_a_new_reply()
    session.credentials = {"github": {"id": "cred-1"}}
    session.successful_agent_runs = {"graph-1": 2}
    session.usage = [
        Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    ]

    await _save_session_to_db(
        session, existing_message_count=1, skip_existence_check=True
    )

    assert [kind for kind, _ in writes] == ["rows"]
    assert writes[0][1]["session_fields"] == ChatSessionSaveFields(
        credentials={"github": {"id": "cred-1"}},
        successful_agent_runs={"graph-1": 2},
        successful_agent_schedules={},
        total_prompt_tokens=105,
        total_completion_tokens=21,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_over_the_database_managers_rpc_the_fields_keep_their_own_write(
    mocker: MockerFixture,
) -> None:
    """During a rolling deploy a database manager from before ``session_fields``
    drops them without an error, so over its RPC they keep their own update."""
    writes = _recording_chat_db(mocker, local=False)
    session = _session_with_a_new_reply()
    session.successful_agent_runs = {"graph-1": 2}

    await _save_session_to_db(
        session, existing_message_count=1, skip_existence_check=True
    )

    assert [kind for kind, _ in writes] == ["session", "rows"]
    assert writes[0][1]["successful_agent_runs"] == {"graph-1": 2}
    assert writes[1][1]["session_fields"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_save_with_nothing_new_still_writes_the_session_fields(
    mocker: MockerFixture,
) -> None:
    writes = _recording_chat_db(mocker, local=True)
    session = _make_session_with_messages(
        ChatMessage(role="user", content="hi", sequence=0)
    )
    session.successful_agent_runs = {"graph-1": 3}

    await _save_session_to_db(
        session, existing_message_count=1, skip_existence_check=True
    )

    assert [kind for kind, _ in writes] == ["session"]
    assert writes[0][1]["successful_agent_runs"] == {"graph-1": 3}
    assert "update_tenancy" not in writes[0][1]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_scope_change_writes_the_session_before_its_new_rows(
    mocker: MockerFixture,
) -> None:
    """A tenancy save keeps its own update, ahead of the rows, as before: the
    fields the rows carry don't include the scope."""
    writes = _recording_chat_db(mocker, local=True)
    session = _session_with_a_new_reply()
    session.organization_id = "personal-org"

    await _save_session_to_db(
        session,
        existing_message_count=1,
        skip_existence_check=True,
        persist_tenancy=True,
    )

    assert [kind for kind, _ in writes] == ["session", "rows"]
    assert writes[0][1]["update_tenancy"] is True
    assert writes[0][1]["organization_id"] == "personal-org"
    assert writes[1][1]["session_fields"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_session_fields_reach_the_db_with_and_without_new_rows(
    setup_test_user, test_user_id, mocker: MockerFixture
):
    """In a process with its own Prisma client, a save with rows writes the
    session's fields in their transaction and one with none writes them on
    their own; a reload from the database sees both."""
    separate_updates = mocker.spy(chat_db_module, "update_chat_session")

    async def reload() -> ChatSession:
        await invalidate_session_cache(s.session_id)
        loaded = await get_chat_session(s.session_id, s.user_id)
        assert loaded is not None
        return loaded

    s = ChatSession.new(user_id=test_user_id, dry_run=False)
    s.messages = [ChatMessage(role="user", content="hi")]
    s = await upsert_chat_session(s)

    s.messages.append(ChatMessage(role="assistant", content="hello"))
    s.successful_agent_runs = {"graph-1": 1}
    s.usage.append(Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
    await upsert_chat_session(s)
    assert separate_updates.call_count == 0
    loaded = await reload()
    assert [m.content for m in loaded.messages] == ["hi", "hello"]
    assert loaded.successful_agent_runs == {"graph-1": 1}
    assert [(u.prompt_tokens, u.completion_tokens) for u in loaded.usage] == [(100, 20)]

    s.successful_agent_schedules = {"graph-2": 1}
    await upsert_chat_session(s)
    assert separate_updates.call_count == 1
    loaded = await reload()
    assert len(loaded.messages) == 2
    assert loaded.successful_agent_schedules == {"graph-2": 1}
    assert loaded.successful_agent_runs == {"graph-1": 1}
