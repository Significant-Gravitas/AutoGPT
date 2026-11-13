"""Integration tests for append_session_message (notes posted between turns)."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage, ChatSession, User

from backend.copilot import db as copilot_db
from backend.copilot.model import get_chat_session
from backend.copilot.scheduled_chats.notices import post_notice, skip_notice
from backend.copilot.turn_queue import TurnRefusal
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
async def test_an_autopilot_schedule_without_a_chat_gets_one_note_chat_per_window(
    server: SpinTestServer,
):
    """A skipped AutoPilot schedule that would have opened a fresh chat has
    no chat of its own yet. Its note opens one, named for the schedule, and a
    second skip in the same window opens nothing more."""
    user_id = f"session-message-{uuid4()}"
    await _create_user(user_id)
    try:
        existing = await _create_session(user_id)
        notice = skip_notice(
            TurnRefusal(reason="paywalled", kind="no_plan"),
            title="Morning digest",
            now=datetime(2026, 10, 19, 9, tzinfo=timezone.utc),
            run="recurring",
        )
        assert notice is not None

        for _ in range(2):
            await post_notice(
                notice,
                user_id=user_id,
                schedule_key="schedule-1",
                title="Morning digest",
                session_id=None,
                expert_id=None,
            )

        chats = await ChatSession.prisma().find_many(where={"userId": user_id})
        opened = [chat for chat in chats if chat.id != existing]
        assert [chat.title for chat in opened] == ["Morning digest"]
        notes = await ChatMessage.prisma().find_many(where={"sessionId": opened[0].id})
        assert [note.content for note in notes] == [notice.content]
        assert await ChatMessage.prisma().count(where={"sessionId": existing}) == 0
    finally:
        await _cleanup(user_id)
