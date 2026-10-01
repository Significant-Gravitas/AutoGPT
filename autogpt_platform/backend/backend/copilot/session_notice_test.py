"""Integration tests for append_session_notice (scheduled follow-up outcomes)."""

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
                "email": f"session-notice-{user_id}@example.com",
                "name": "Session Notice Test",
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
async def test_append_session_notice_lands_in_the_named_session_and_dedupes(
    server: SpinTestServer,
):
    user_id = f"session-notice-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await _create_session(user_id)
        message_id = str(uuid4())
        metadata = {"kind": "scheduled_followup_outcome", "status": "dropped"}

        assert await copilot_db.append_session_notice(
            session_id=session_id,
            user_id=user_id,
            content="The follow-up did not run.",
            message_id=message_id,
            metadata=metadata,
        )
        # A double fire with the same deterministic id posts nothing more.
        assert not await copilot_db.append_session_notice(
            session_id=session_id,
            user_id=user_id,
            content="The follow-up did not run.",
            message_id=message_id,
        )

        rows = await ChatMessage.prisma().find_many(where={"sessionId": session_id})
        assert len(rows) == 1
        assert rows[0].id == message_id
        assert rows[0].role == "assistant"
        assert rows[0].content == "The follow-up did not run."
        assert rows[0].metadata == metadata

        # The next turn reads the session back with the notice in it.
        session = await get_chat_session(session_id, user_id)
        assert session is not None
        assert [m.content for m in session.messages] == ["The follow-up did not run."]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_append_session_notice_refuses_a_session_the_user_does_not_own(
    server: SpinTestServer,
):
    """Unlike the expert/plain posters this never creates or re-targets a
    session: a notice about a chat belongs in that chat or nowhere."""
    owner = f"session-notice-owner-{uuid4()}"
    other = f"session-notice-other-{uuid4()}"
    await _create_user(owner)
    await _create_user(other)
    try:
        session_id = await _create_session(owner)
        assert not await copilot_db.append_session_notice(
            session_id=session_id,
            user_id=other,
            content="nope",
            message_id=str(uuid4()),
        )
        assert not await copilot_db.append_session_notice(
            session_id=str(uuid4()),
            user_id=owner,
            content="nope",
            message_id=str(uuid4()),
        )
        assert await ChatMessage.prisma().count(where={"sessionId": session_id}) == 0
    finally:
        await _cleanup(owner)
        await _cleanup(other)
