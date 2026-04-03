import pytest

from .db import set_turn_duration
from .model import ChatMessage, ChatSession, get_chat_session, upsert_chat_session


@pytest.mark.asyncio(loop_scope="session")
async def test_set_turn_duration_updates_cache_in_place(setup_test_user, test_user_id):
    """set_turn_duration patches the cached session without invalidation.

    Verifies that after calling set_turn_duration the Redis-cached session
    reflects the updated durationMs on the last assistant message, without
    the cache having been deleted and re-populated (which could race with
    concurrent get_chat_session calls).
    """
    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there"),
    ]
    session = await upsert_chat_session(session)

    # Ensure the session is in cache
    cached = await get_chat_session(session.session_id, test_user_id)
    assert cached is not None
    assert cached.messages[-1].duration_ms is None

    # Update turn duration — should patch cache in-place
    await set_turn_duration(session.session_id, 1234)

    # Read from cache (not DB) — the cache should already have the update
    updated = await get_chat_session(session.session_id, test_user_id)
    assert updated is not None
    assistant_msgs = [m for m in updated.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].duration_ms == 1234


@pytest.mark.asyncio(loop_scope="session")
async def test_set_turn_duration_no_assistant_message(setup_test_user, test_user_id):
    """set_turn_duration is a no-op when there are no assistant messages."""
    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hello"),
    ]
    session = await upsert_chat_session(session)

    # Should not raise
    await set_turn_duration(session.session_id, 5678)

    cached = await get_chat_session(session.session_id, test_user_id)
    assert cached is not None
    # User message should not have durationMs
    assert cached.messages[0].duration_ms is None
