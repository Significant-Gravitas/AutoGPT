"""Integration tests for append_plain_session_message."""

import logging
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage, Expert, User

from backend.copilot import db as copilot_db
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _create_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"plain-session-{user_id}@example.com",
                "name": "Plain Session Test",
            }
        )
    except UniqueViolationError:
        pass


async def _cleanup(user_id: str) -> None:
    try:
        # ChatSession -> User, ChatMessage -> ChatSession, and Expert.ownerUserId
        # -> User are all onDelete: Cascade, so deleting the user sweeps
        # everything created for it, including any expert + expert-scoped
        # session set up for the discrimination check below.
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_append_plain_session_message_creates_session_and_dedupes(
    server: SpinTestServer,
):
    user_id = f"plain-session-{uuid4()}"
    await _create_user(user_id)
    try:
        message_id = str(uuid4())

        session_id = await copilot_db.append_plain_session_message(
            user_id=user_id,
            content="## Briefing",
            message_id=message_id,
            metadata={"kind": "morning_briefing"},
        )
        assert session_id is not None

        # Same message id -> dedup, no second message.
        assert (
            await copilot_db.append_plain_session_message(
                user_id=user_id, content="## Briefing", message_id=message_id
            )
            is None
        )

        # Reuses the latest plain session instead of creating another.
        second = await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing 2", message_id=str(uuid4())
        )
        assert second == session_id

        # An expert-scoped session that is the MOST RECENTLY updated session
        # for this user must still be skipped -- the {"expertId": None}
        # filter has to actively discriminate, not just "reuse whatever's
        # newest." Without the filter this would wrongly post into the
        # expert's thread.
        expert = await Expert.prisma().create(
            data={
                "ownerUserId": user_id,
                "name": "Test Expert",
                "role": "assistant",
                "identity": "test",
            }
        )
        expert_session = await copilot_db.create_chat_session(
            session_id=str(uuid4()), user_id=user_id, expert_id=expert.id
        )
        assert expert_session.session_id != session_id

        third = await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing 3", message_id=str(uuid4())
        )
        assert third == session_id
        assert third != expert_session.session_id

        expert_messages = await ChatMessage.prisma().find_many(
            where={"sessionId": expert_session.session_id}
        )
        assert expert_messages == []
    finally:
        await _cleanup(user_id)
