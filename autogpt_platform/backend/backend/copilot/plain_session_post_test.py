"""Integration tests for briefing chat creation and delivery retries."""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import Expert, User

from backend.copilot import db as copilot_db
from backend.copilot.model import ChatSession, ChatSessionMetadata
from backend.copilot.tools.expert_proposal import autopilot_session_guard
from backend.util.test import SpinTestServer


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server: SpinTestServer):
    user_id = f"plain-session-{uuid4()}"
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    try:
        yield user_id
    finally:
        await User.prisma().delete_many(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_briefing_leaves_existing_conversations_unchanged(user_id: str):
    existing_ids = await _seed_existing_conversations(user_id)
    sessions_before = await PrismaChatSession.prisma().find_many(
        where={"id": {"in": existing_ids}}, order={"id": "asc"}
    )
    messages_before = await ChatMessage.prisma().find_many(
        where={"sessionId": {"in": existing_ids}}, order={"id": "asc"}
    )
    message_id = str(uuid4())
    metadata = {"kind": "morning_briefing", "briefing_id": "briefing-1"}

    session_id = await copilot_db.append_plain_session_message(
        user_id=user_id,
        content="## Briefing",
        message_id=message_id,
        metadata=metadata,
    )

    assert session_id is not None and session_id not in existing_ids
    created = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id}, include={"Messages": True}
    )
    assert created is not None and created.expertId is None
    assert created.userId == user_id
    assert created.Messages is not None and len(created.Messages) == 1
    message = created.Messages[0]
    assert (message.id, message.role, message.sequence) == (message_id, "assistant", 0)
    assert message.content == "## Briefing"
    assert message.metadata == metadata
    assert (
        await PrismaChatSession.prisma().find_many(
            where={"id": {"in": existing_ids}}, order={"id": "asc"}
        )
        == sessions_before
    )
    assert (
        await ChatMessage.prisma().find_many(
            where={"sessionId": {"in": existing_ids}}, order={"id": "asc"}
        )
        == messages_before
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_distinct_briefings_create_separate_chats_and_retries_dedupe(
    user_id: str,
):
    first_message_id = str(uuid4())
    first = await copilot_db.append_plain_session_message(
        user_id=user_id, content="## Briefing", message_id=first_message_id
    )
    second = await copilot_db.append_plain_session_message(
        user_id=user_id, content="## Next briefing", message_id=str(uuid4())
    )
    assert first is not None and second is not None and first != second

    assert (
        await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing", message_id=first_message_id
        )
        is None
    )
    sessions = await PrismaChatSession.prisma().find_many(
        where={"userId": user_id}, include={"Messages": True}
    )
    assert {session.id for session in sessions} == {first, second}
    assert all(session.Messages and len(session.Messages) == 1 for session in sessions)


@pytest.mark.asyncio(loop_scope="session")
async def test_briefing_chat_preserves_interactive_origin_and_default_route(
    user_id: str, mocker
):
    route = mocker.patch.object(
        copilot_db,
        "resolve_default_chat_route",
        AsyncMock(return_value=("codex", "user-codex-credential")),
    )
    session_id = await copilot_db.append_plain_session_message(
        user_id=user_id, content="## Briefing", message_id=str(uuid4())
    )
    assert session_id is not None
    created = await copilot_db.get_chat_session_metadata(session_id)
    assert created is not None
    assert created.metadata.origin == "interactive"
    assert created.metadata.kind == "normal"
    assert created.metadata.builder_graph_id is None
    assert created.metadata.dry_run is False
    assert created.metadata.llm_auth_provider == "codex"
    assert created.metadata.llm_credential_id == "user-codex-credential"
    route.assert_awaited_once_with(user_id)
    session = ChatSession.new(user_id, dry_run=False, session_id=session_id)
    session.metadata = created.metadata
    assert autopilot_session_guard(user_id, session) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_briefing_chat_has_an_initial_title(user_id: str):
    session_id = await copilot_db.append_plain_session_message(
        user_id=user_id,
        content="## Briefing",
        message_id=str(uuid4()),
        title="Morning briefing — 2026-10-07",
    )
    assert session_id is not None
    created = await copilot_db.get_chat_session_metadata(session_id)
    assert created is not None
    assert created.title == "Morning briefing — 2026-10-07"


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_deliveries_create_one_chat_without_orphans(
    user_id: str, mocker
):
    existing = await copilot_db.create_chat_session(
        session_id=str(uuid4()), user_id=user_id
    )
    message_actions = ChatMessage.prisma()
    real_find_unique = message_actions.find_unique
    barrier = asyncio.Barrier(2)
    arrivals = 0

    async def race_precheck(**kwargs):
        nonlocal arrivals
        arrivals += 1
        if arrivals <= 2:
            await asyncio.wait_for(barrier.wait(), timeout=10)
            return None
        return await real_find_unique(**kwargs)

    mocker.patch.object(
        type(message_actions), "find_unique", AsyncMock(side_effect=race_precheck)
    )
    message_id = str(uuid4())
    results = await asyncio.gather(
        *[
            copilot_db.append_plain_session_message(
                user_id=user_id, content="## Briefing", message_id=message_id
            )
            for _ in range(2)
        ]
    )
    assert results.count(None) == 1
    delivered = next(result for result in results if result is not None)
    assert delivered != existing.session_id
    sessions = await PrismaChatSession.prisma().find_many(
        where={"userId": user_id}, include={"Messages": True}
    )
    assert {session.id for session in sessions} == {existing.session_id, delivered}
    assert sum(len(session.Messages or []) for session in sessions) == 1
    original = next(
        session for session in sessions if session.id == existing.session_id
    )
    assert original.Messages == []


@pytest.mark.parametrize("unique_failure", [False, True])
@pytest.mark.asyncio(loop_scope="session")
async def test_chat_creation_failures_propagate_without_leaving_a_session(
    user_id: str, mocker, unique_failure: bool
):
    error = (
        UniqueViolationError(
            {"user_facing_error": {"message": "Unique constraint: ChatSession_pkey"}}
        )
        if unique_failure
        else RuntimeError("chat creation failed")
    )
    session_actions = PrismaChatSession.prisma()
    mocker.patch.object(type(session_actions), "create", AsyncMock(side_effect=error))

    with pytest.raises(type(error)):
        await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing", message_id=str(uuid4())
        )
    assert await PrismaChatSession.prisma().find_many(where={"userId": user_id}) == []


async def _seed_existing_conversations(user_id: str) -> list[str]:
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Test Expert",
            "role": "assistant",
            "identity": "test",
        }
    )
    sessions = [
        await copilot_db.create_chat_session(
            session_id=str(uuid4()), user_id=user_id, metadata=metadata
        )
        for metadata in (
            ChatSessionMetadata(origin="interactive"),
            ChatSessionMetadata(
                origin="interactive", builder_graph_id="existing-agent"
            ),
            ChatSessionMetadata(kind="dream"),
        )
    ]
    sessions.append(
        await copilot_db.create_chat_session(
            session_id=str(uuid4()), user_id=user_id, expert_id=expert.id
        )
    )
    for session in sessions:
        await copilot_db.add_chat_message(
            session_id=session.session_id,
            role="assistant",
            sequence=0,
            content="Existing conversation",
        )
    return [session.session_id for session in sessions]
