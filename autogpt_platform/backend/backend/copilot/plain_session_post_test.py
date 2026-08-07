"""Integration tests for append_plain_session_message."""

import logging
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import User

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
        # ChatSession -> User and ChatMessage -> ChatSession are both
        # onDelete: Cascade, so deleting the user sweeps everything.
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
    finally:
        await _cleanup(user_id)
