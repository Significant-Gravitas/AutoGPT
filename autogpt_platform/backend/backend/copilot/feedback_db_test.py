"""Tests for ``copilot.feedback_db`` against the real Postgres.

The owner predicate is the whole authorization story for a rating, so it is
proven against the database rather than a mocked query.
"""

from uuid import uuid4

import pytest
from prisma.models import ChatMessageFeedback

from backend.copilot.db import add_chat_message, create_chat_session
from backend.copilot.feedback_db import get_rateable_message, upsert_message_feedback


async def _chat_with_reply(user_id: str) -> tuple[str, str]:
    """A session owned by *user_id* holding a prompt (seq 0) and a reply
    (seq 1). Returns the session id and the reply's row id."""
    session = await create_chat_session(str(uuid4()), user_id)
    await add_chat_message(session.session_id, "user", 0, content="Summarise it")
    reply = await add_chat_message(
        session.session_id, "assistant", 1, content="Here is the summary."
    )
    assert reply.id is not None
    return session.session_id, reply.id


@pytest.mark.asyncio(loop_scope="session")
async def test_reply_is_found_only_through_its_owners_session(
    setup_test_user, test_user_id, setup_admin_user, admin_user_id
):
    session_id, reply_id = await _chat_with_reply(test_user_id)

    by_sequence = await get_rateable_message(test_user_id, session_id, sequence=1)
    by_id = await get_rateable_message(test_user_id, session_id, message_id=reply_id)
    assert by_sequence is not None and by_sequence.id == reply_id
    assert by_id is not None and by_id.id == reply_id

    other_user = admin_user_id
    assert await get_rateable_message(other_user, session_id, sequence=1) is None
    assert (
        await get_rateable_message(other_user, session_id, message_id=reply_id) is None
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_only_replies_in_the_named_session_are_rateable(
    setup_test_user, test_user_id
):
    session_id, reply_id = await _chat_with_reply(test_user_id)
    other_session_id, _ = await _chat_with_reply(test_user_id)

    # The prompt is the user's own words, not a reply.
    assert await get_rateable_message(test_user_id, session_id, sequence=0) is None
    assert await get_rateable_message(test_user_id, session_id, sequence=99) is None
    # The same user's reply, named against a different chat.
    assert (
        await get_rateable_message(test_user_id, other_session_id, message_id=reply_id)
        is None
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_rating_again_updates_the_same_row(setup_test_user, test_user_id):
    session_id, reply_id = await _chat_with_reply(test_user_id)

    first = await upsert_message_feedback(
        user_id=test_user_id,
        session_id=session_id,
        message_id=reply_id,
        score_name="user-feedback",
        score_value=0,
        comment="Wrong numbers",
        langfuse_trace_id=None,
    )
    second = await upsert_message_feedback(
        user_id=test_user_id,
        session_id=session_id,
        message_id=reply_id,
        score_name="user-feedback",
        score_value=1,
        comment=None,
        langfuse_trace_id="1edf31f11b1693cc6103f358c1481694",
    )

    assert second == first
    rows = await ChatMessageFeedback.prisma().find_many(where={"messageId": reply_id})
    assert len(rows) == 1
    assert rows[0].scoreValue == 1
    assert rows[0].comment is None
    assert rows[0].langfuseTraceId == "1edf31f11b1693cc6103f358c1481694"
    assert rows[0].sessionId == session_id
    assert rows[0].userId == test_user_id


@pytest.mark.asyncio(loop_scope="session")
async def test_each_score_name_keeps_its_own_rating(setup_test_user, test_user_id):
    session_id, reply_id = await _chat_with_reply(test_user_id)

    vote = await upsert_message_feedback(
        user_id=test_user_id,
        session_id=session_id,
        message_id=reply_id,
        score_name="user-feedback",
        score_value=1,
        comment=None,
        langfuse_trace_id=None,
    )
    copy = await upsert_message_feedback(
        user_id=test_user_id,
        session_id=session_id,
        message_id=reply_id,
        score_name="copy",
        score_value=1,
        comment=None,
        langfuse_trace_id=None,
    )

    assert vote != copy


@pytest.mark.asyncio(loop_scope="session")
async def test_comment_is_stripped_of_postgres_control_characters(
    setup_test_user, test_user_id
):
    session_id, reply_id = await _chat_with_reply(test_user_id)

    feedback_id = await upsert_message_feedback(
        user_id=test_user_id,
        session_id=session_id,
        message_id=reply_id,
        score_name="user-feedback",
        score_value=0,
        comment="bad\x00 answer",
        langfuse_trace_id=None,
    )

    row = await ChatMessageFeedback.prisma().find_unique(where={"id": feedback_id})
    assert row is not None
    assert row.comment == "bad answer"
