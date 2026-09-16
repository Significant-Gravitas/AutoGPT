"""Integration tests for append_plain_session_message."""

import logging
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage, Expert, User

from backend.copilot import db as copilot_db
from backend.copilot.model import ChatSession, ChatSessionMetadata
from backend.copilot.tools.expert_proposal import autopilot_session_guard
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


@pytest.mark.asyncio(loop_scope="session")
async def test_append_plain_session_message_creates_an_interactive_thread(
    server: SpinTestServer,
):
    """The auto-created primary thread must not be stamped as an automation.

    A new user in the experts cohort gets a briefing before their first chat
    and is dropped straight into this thread. Inheriting the unset-metadata
    ``automation`` default would have ``autopilot_session_guard`` tell them
    their own thread "was started by an automation" the moment they ask to
    hire someone.
    """
    user_id = f"plain-session-origin-{uuid4()}"
    await _create_user(user_id)
    try:
        session_id = await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing", message_id=str(uuid4())
        )
        assert session_id is not None

        created = await copilot_db.get_chat_session_metadata(session_id)
        assert created is not None
        assert created.metadata.origin == "interactive"

        session = ChatSession.new(user_id, dry_run=False, session_id=session_id)
        session.metadata = created.metadata
        assert autopilot_session_guard(user_id, session) is None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_append_plain_session_message_retries_on_sequence_collision(
    server: SpinTestServer, mocker
):
    """Lock-degraded path: a sequence PK collision (not a duplicate message
    id) must be retried once with a fresh sequence rather than propagating."""
    user_id = f"plain-session-retry-{uuid4()}"
    await _create_user(user_id)
    try:
        real_add_chat_message = copilot_db.add_chat_message
        calls = {"n": 0}

        async def collide_once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # A (sessionId, sequence) collision, NOT a duplicate
                # ChatMessage id — the branch under test.
                raise UniqueViolationError(
                    {
                        "user_facing_error": {
                            "message": "Unique constraint failed on the "
                            "fields: (`sessionId`,`sequence`)"
                        }
                    }
                )
            return await real_add_chat_message(*args, **kwargs)

        mocker.patch.object(copilot_db, "add_chat_message", side_effect=collide_once)

        message_id = str(uuid4())
        session_id = await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing", message_id=message_id
        )

        assert calls["n"] == 2  # first write collided, retry succeeded
        assert session_id is not None
        stored = await ChatMessage.prisma().find_unique(where={"id": message_id})
        assert stored is not None
        assert stored.sessionId == session_id
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_append_plain_session_message_skips_dream_sessions(
    server: SpinTestServer,
):
    """Dream sessions are also ``expertId IS NULL`` but are hidden from every
    listing surface, so a briefing posted into one is invisible forever — and
    still gets stamped delivered."""
    user_id = f"plain-session-dream-{uuid4()}"
    await _create_user(user_id)
    try:
        plain = await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing", message_id=str(uuid4())
        )
        assert plain is not None

        # Created after the plain session, so it is the most recently updated
        # expertId-IS-NULL session for this user.
        dream_session = await copilot_db.create_chat_session(
            session_id=str(uuid4()),
            user_id=user_id,
            metadata=ChatSessionMetadata(kind="dream"),
        )
        assert dream_session.session_id != plain

        landed = await copilot_db.append_plain_session_message(
            user_id=user_id, content="## Briefing 2", message_id=str(uuid4())
        )

        assert landed == plain
        dream_messages = await ChatMessage.prisma().find_many(
            where={"sessionId": dream_session.session_id}
        )
        assert dream_messages == []
    finally:
        await _cleanup(user_id)
