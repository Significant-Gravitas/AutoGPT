"""Integration tests for append_session_message (notes posted between turns)."""

import logging
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage, User

from backend.copilot import db as copilot_db
from backend.copilot.model import get_chat_session
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _create_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"session-message-{user_id}@example.com",
                "name": "Session Message Test",
            }
        )
    except UniqueViolationError:
        pass


async def _create_session(user_id: str) -> str:
    created = await copilot_db.create_chat_session(
        session_id=str(uuid4()), user_id=user_id
    )
    return created.session_id


async def _cleanup(user_id: str) -> None:
    try:
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_note_lands_in_the_named_chat_once(server: SpinTestServer):
    user_id = f"session-message-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _create_session(user_id)
        message_id = str(uuid4())
        metadata = {"kind": "scheduled_chat_notice", "hidden_from_model": True}

        assert await copilot_db.append_session_message(
            session_id=session_id,
            user_id=user_id,
            content="Skipped: daily usage limit reached.",
            message_id=message_id,
            metadata=metadata,
        )
        # A second skip in the same window carries the same id: nothing more.
        assert not await copilot_db.append_session_message(
            session_id=session_id,
            user_id=user_id,
            content="Skipped: daily usage limit reached.",
            message_id=message_id,
            metadata=metadata,
        )

        rows = await ChatMessage.prisma().find_many(where={"sessionId": session_id})
        assert [(r.id, r.role, r.metadata) for r in rows] == [
            (message_id, "assistant", metadata)
        ]
        # The chat page reads the session back with the note in it, not a
        # cached copy from before it was written.
        session = await get_chat_session(session_id, user_id)
        assert session is not None
        assert [m.content for m in session.messages] == [
            "Skipped: daily usage limit reached."
        ]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_note_never_lands_in_someone_elses_chat(server: SpinTestServer):
    """A note about a chat belongs in that chat or nowhere: no fallback, no
    new session."""
    owner = f"session-message-owner-{uuid4()}"
    other = f"session-message-other-{uuid4()}"
    await _create_user(owner)
    await _create_user(other)
    try:
        session_id = await _create_session(owner)
        assert not await copilot_db.append_session_message(
            session_id=session_id,
            user_id=other,
            content="nope",
            message_id=str(uuid4()),
        )
        assert not await copilot_db.append_session_message(
            session_id=str(uuid4()),
            user_id=owner,
            content="nope",
            message_id=str(uuid4()),
        )
        assert await ChatMessage.prisma().count(where={"sessionId": session_id}) == 0
    finally:
        await _cleanup(owner)
        await _cleanup(other)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_note_in_the_main_chat_reaches_a_copy_already_cached(
    server: SpinTestServer,
):
    """A skipped AutoPilot schedule with no chat of its own posts into the
    owner's main chat. Whoever had that chat open has it in the cache."""
    user_id = f"session-message-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _create_session(user_id)
        cached = await get_chat_session(session_id, user_id)
        assert cached is not None and cached.messages == []

        posted_in = await copilot_db.append_plain_session_message(
            user_id=user_id,
            content="Skipped: scheduled chats need a plan",
            message_id=str(uuid4()),
            metadata={"kind": "scheduled_chat_notice", "hidden_from_model": True},
        )

        assert posted_in == session_id
        session = await get_chat_session(session_id, user_id)
        assert session is not None
        assert [m.content for m in session.messages] == [
            "Skipped: scheduled chats need a plan"
        ]
    finally:
        await _cleanup(user_id)
