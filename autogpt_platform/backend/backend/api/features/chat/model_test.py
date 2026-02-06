from unittest.mock import patch

import pytest
from prisma.errors import UniqueViolationError

from .model import (
    ChatMessage,
    ChatSession,
    Usage,
    get_chat_session,
    upsert_chat_session,
)

messages = [
    ChatMessage(content="Hello, how are you?", role="user"),
    ChatMessage(
        content="I'm fine, thank you!",
        role="assistant",
        tool_calls=[
            {
                "id": "t123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "New York"}',
                },
            }
        ],
    ),
    ChatMessage(
        content="I'm using the tool to get the weather",
        role="tool",
        tool_call_id="t123",
    ),
]


@pytest.mark.asyncio(loop_scope="session")
async def test_chatsession_serialization_deserialization():
    s = ChatSession.new(user_id="abc123")
    s.messages = messages
    s.usage = [Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)]
    serialized = s.model_dump_json()
    s2 = ChatSession.model_validate_json(serialized)
    assert s2.model_dump() == s.model_dump()


@pytest.mark.asyncio(loop_scope="session")
async def test_chatsession_redis_storage(setup_test_user, test_user_id):

    s = ChatSession.new(user_id=test_user_id)
    s.messages = messages

    s = await upsert_chat_session(s)

    s2 = await get_chat_session(
        session_id=s.session_id,
        user_id=s.user_id,
    )

    assert s2 == s


@pytest.mark.asyncio(loop_scope="session")
async def test_chatsession_redis_storage_user_id_mismatch(
    setup_test_user, test_user_id
):

    s = ChatSession.new(user_id=test_user_id)
    s.messages = messages
    s = await upsert_chat_session(s)

    s2 = await get_chat_session(s.session_id, "different_user_id")

    assert s2 is None


@pytest.mark.asyncio(loop_scope="session")
async def test_chatsession_db_storage(setup_test_user, test_user_id):
    """Test that messages are correctly saved to and loaded from DB (not cache)."""
    from backend.data.redis_client import get_redis_async

    # Create session with messages including assistant message
    s = ChatSession.new(user_id=test_user_id)
    s.messages = messages  # Contains user, assistant, and tool messages
    assert s.session_id is not None, "Session id is not set"
    # Upsert to save to both cache and DB
    s = await upsert_chat_session(s)

    # Clear the Redis cache to force DB load
    redis_key = f"chat:session:{s.session_id}"
    async_redis = await get_redis_async()
    await async_redis.delete(redis_key)

    # Load from DB (cache was cleared)
    s2 = await get_chat_session(
        session_id=s.session_id,
        user_id=s.user_id,
    )

    assert s2 is not None, "Session not found after loading from DB"
    assert len(s2.messages) == len(
        s.messages
    ), f"Message count mismatch: expected {len(s.messages)}, got {len(s2.messages)}"

    # Verify all roles are present
    roles = [m.role for m in s2.messages]
    assert "user" in roles, f"User message missing. Roles found: {roles}"
    assert "assistant" in roles, f"Assistant message missing. Roles found: {roles}"
    assert "tool" in roles, f"Tool message missing. Roles found: {roles}"

    # Verify message content
    for orig, loaded in zip(s.messages, s2.messages):
        assert orig.role == loaded.role, f"Role mismatch: {orig.role} != {loaded.role}"
        assert (
            orig.content == loaded.content
        ), f"Content mismatch for {orig.role}: {orig.content} != {loaded.content}"
        if orig.tool_calls:
            assert (
                loaded.tool_calls is not None
            ), f"Tool calls missing for {orig.role} message"
            assert len(orig.tool_calls) == len(loaded.tool_calls)


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_handles_concurrent_saves(setup_test_user, test_user_id):
    """Test that incremental saves work: save initial messages, add more, save again."""
    from backend.data.redis_client import get_redis_async

    # Create session with initial messages
    s = ChatSession.new(user_id=test_user_id)
    s.messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="First reply", role="assistant"),
    ]
    s = await upsert_chat_session(s)

    # Add more messages and save again (incremental)
    s.messages.append(ChatMessage(content="Second message", role="user"))
    s.messages.append(ChatMessage(content="Second reply", role="assistant"))
    s = await upsert_chat_session(s)

    # Clear cache and verify all messages round-trip from DB
    redis_key = f"chat:session:{s.session_id}"
    async_redis = await get_redis_async()
    await async_redis.delete(redis_key)

    loaded = await get_chat_session(session_id=s.session_id, user_id=s.user_id)
    assert loaded is not None, "Session not found after incremental save"
    assert len(loaded.messages) == 4, f"Expected 4 messages, got {len(loaded.messages)}"

    # Verify content of all messages
    assert loaded.messages[0].content == "First message"
    assert loaded.messages[1].content == "First reply"
    assert loaded.messages[2].content == "Second message"
    assert loaded.messages[3].content == "Second reply"


@pytest.mark.asyncio(loop_scope="session")
async def test_upsert_retries_on_unique_violation(setup_test_user, test_user_id):
    """Test that upsert_chat_session retries when UniqueViolationError is raised."""
    from . import db as chat_db

    # Create a session with initial messages
    s = ChatSession.new(user_id=test_user_id)
    s.messages = [
        ChatMessage(content="Hello", role="user"),
    ]
    s = await upsert_chat_session(s)

    # Add a new message
    s.messages.append(ChatMessage(content="World", role="assistant"))

    # Mock add_chat_messages_batch to raise UniqueViolationError on first call,
    # then succeed on second call (simulating another pod saving concurrently).
    original_batch = chat_db.add_chat_messages_batch
    call_count = 0

    async def mock_batch(session_id, messages, start_sequence):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise UniqueViolationError(
                {
                    "error": "Unique constraint failed on the fields: (sessionId, sequence)"
                }
            )
        return await original_batch(session_id, messages, start_sequence)

    with patch.object(chat_db, "add_chat_messages_batch", side_effect=mock_batch):
        # Also mock get_chat_session_message_count to return 1 on retry
        # (simulating that the first message was saved by "another pod")
        original_count = chat_db.get_chat_session_message_count

        count_call = 0

        async def mock_count(session_id):
            nonlocal count_call
            count_call += 1
            # First call is the initial count check in upsert_chat_session
            # Second call is the retry after UniqueViolationError
            return await original_count(session_id)

        with patch.object(
            chat_db, "get_chat_session_message_count", side_effect=mock_count
        ):
            s = await upsert_chat_session(s)

    # Verify the session completed without error
    assert s is not None
    assert len(s.messages) == 2
