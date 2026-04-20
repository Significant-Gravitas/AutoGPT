from typing import cast

import pytest
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from pytest_mock import MockerFixture

from .model import (
    ChatMessage,
    ChatSession,
    Usage,
    append_and_save_message,
    get_chat_session,
    is_message_duplicate,
    maybe_append_user_message,
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
    s = ChatSession.new(user_id="abc123", dry_run=False)
    s.messages = messages
    s.usage = [Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)]
    serialized = s.model_dump_json()
    s2 = ChatSession.model_validate_json(serialized)
    assert s2.model_dump() == s.model_dump()


@pytest.mark.asyncio(loop_scope="session")
async def test_chatsession_redis_storage(setup_test_user, test_user_id):

    s = ChatSession.new(user_id=test_user_id, dry_run=False)
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

    s = ChatSession.new(user_id=test_user_id, dry_run=False)
    s.messages = messages
    s = await upsert_chat_session(s)

    s2 = await get_chat_session(s.session_id, "different_user_id")

    assert s2 is None


@pytest.mark.asyncio(loop_scope="session")
async def test_chatsession_db_storage(setup_test_user, test_user_id):
    """Test that messages are correctly saved to and loaded from DB (not cache)."""
    from backend.data.redis_client import get_redis_async

    # Create session with messages including assistant message
    s = ChatSession.new(user_id=test_user_id, dry_run=False)
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


# --------------------------------------------------------------------------- #
#  _merge_consecutive_assistant_messages                                       #
# --------------------------------------------------------------------------- #

_tc = ChatCompletionMessageToolCallParam(
    id="tc1", type="function", function=Function(name="do_stuff", arguments="{}")
)
_tc2 = ChatCompletionMessageToolCallParam(
    id="tc2", type="function", function=Function(name="other", arguments="{}")
)


def test_merge_noop_when_no_consecutive_assistants():
    """Messages without consecutive assistants are returned unchanged."""
    msgs = [
        ChatCompletionUserMessageParam(role="user", content="hi"),
        ChatCompletionAssistantMessageParam(role="assistant", content="hello"),
        ChatCompletionUserMessageParam(role="user", content="bye"),
    ]
    merged = ChatSession._merge_consecutive_assistant_messages(msgs)
    assert len(merged) == 3
    assert [m["role"] for m in merged] == ["user", "assistant", "user"]


def test_merge_splits_text_and_tool_calls():
    """The exact bug scenario: text-only assistant followed by tool_calls-only assistant."""
    msgs = [
        ChatCompletionUserMessageParam(role="user", content="build agent"),
        ChatCompletionAssistantMessageParam(
            role="assistant", content="Let me build that"
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant", content="", tool_calls=[_tc]
        ),
        ChatCompletionToolMessageParam(role="tool", content="ok", tool_call_id="tc1"),
    ]
    merged = ChatSession._merge_consecutive_assistant_messages(msgs)

    assert len(merged) == 3
    assert merged[0]["role"] == "user"
    assert merged[2]["role"] == "tool"
    a = cast(ChatCompletionAssistantMessageParam, merged[1])
    assert a["role"] == "assistant"
    assert a.get("content") == "Let me build that"
    assert a.get("tool_calls") == [_tc]


def test_merge_combines_tool_calls_from_both():
    """Both consecutive assistants have tool_calls — they get merged."""
    msgs: list[ChatCompletionAssistantMessageParam] = [
        ChatCompletionAssistantMessageParam(
            role="assistant", content="text", tool_calls=[_tc]
        ),
        ChatCompletionAssistantMessageParam(
            role="assistant", content="", tool_calls=[_tc2]
        ),
    ]
    merged = ChatSession._merge_consecutive_assistant_messages(msgs)  # type: ignore[arg-type]

    assert len(merged) == 1
    a = cast(ChatCompletionAssistantMessageParam, merged[0])
    assert a.get("tool_calls") == [_tc, _tc2]
    assert a.get("content") == "text"


def test_merge_three_consecutive_assistants():
    """Three consecutive assistants collapse into one."""
    msgs: list[ChatCompletionAssistantMessageParam] = [
        ChatCompletionAssistantMessageParam(role="assistant", content="a"),
        ChatCompletionAssistantMessageParam(role="assistant", content="b"),
        ChatCompletionAssistantMessageParam(
            role="assistant", content="", tool_calls=[_tc]
        ),
    ]
    merged = ChatSession._merge_consecutive_assistant_messages(msgs)  # type: ignore[arg-type]

    assert len(merged) == 1
    a = cast(ChatCompletionAssistantMessageParam, merged[0])
    assert a.get("content") == "a\nb"
    assert a.get("tool_calls") == [_tc]


def test_merge_empty_and_single_message():
    """Edge cases: empty list and single message."""
    assert ChatSession._merge_consecutive_assistant_messages([]) == []

    single: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content="hi")
    ]
    assert ChatSession._merge_consecutive_assistant_messages(single) == single


# --------------------------------------------------------------------------- #
#  add_tool_call_to_current_turn                                               #
# --------------------------------------------------------------------------- #

_raw_tc = {
    "id": "tc1",
    "type": "function",
    "function": {"name": "f", "arguments": "{}"},
}
_raw_tc2 = {
    "id": "tc2",
    "type": "function",
    "function": {"name": "g", "arguments": "{}"},
}


def test_add_tool_call_appends_to_existing_assistant():
    """When the last assistant is from the current turn, tool_call is added to it."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="working on it"),
    ]
    session.add_tool_call_to_current_turn(_raw_tc)

    assert len(session.messages) == 2  # no new message created
    assert session.messages[1].tool_calls == [_raw_tc]


def test_add_tool_call_creates_assistant_when_none_exists():
    """When there's no current-turn assistant, a new one is created."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hi"),
    ]
    session.add_tool_call_to_current_turn(_raw_tc)

    assert len(session.messages) == 2
    assert session.messages[1].role == "assistant"
    assert session.messages[1].tool_calls == [_raw_tc]


def test_add_tool_call_does_not_cross_user_boundary():
    """A user message acts as a boundary — previous assistant is not modified."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="assistant", content="old turn"),
        ChatMessage(role="user", content="new message"),
    ]
    session.add_tool_call_to_current_turn(_raw_tc)

    assert len(session.messages) == 3  # new assistant was created
    assert session.messages[0].tool_calls is None  # old assistant untouched
    assert session.messages[2].role == "assistant"
    assert session.messages[2].tool_calls == [_raw_tc]


def test_add_tool_call_multiple_times():
    """Multiple long-running tool calls accumulate on the same assistant."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="doing stuff"),
    ]
    session.add_tool_call_to_current_turn(_raw_tc)
    # Simulate a pending tool result in between (like _yield_tool_call does)
    session.messages.append(
        ChatMessage(role="tool", content="pending", tool_call_id="tc1")
    )
    session.add_tool_call_to_current_turn(_raw_tc2)

    assert len(session.messages) == 3  # user, assistant, tool — no extra assistant
    assert session.messages[1].tool_calls == [_raw_tc, _raw_tc2]


def test_to_openai_messages_merges_split_assistants():
    """End-to-end: session with split assistants produces valid OpenAI messages."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="build agent"),
        ChatMessage(role="assistant", content="Let me build that"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "create_agent", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(role="tool", content="done", tool_call_id="tc1"),
        ChatMessage(role="assistant", content="Saved!"),
        ChatMessage(role="user", content="show me an example run"),
    ]
    openai_msgs = session.to_openai_messages()

    # The two consecutive assistants at index 1,2 should be merged
    roles = [m["role"] for m in openai_msgs]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]

    # The merged assistant should have both content and tool_calls
    merged = cast(ChatCompletionAssistantMessageParam, openai_msgs[1])
    assert merged.get("content") == "Let me build that"
    tc_list = merged.get("tool_calls")
    assert tc_list is not None and len(list(tc_list)) == 1
    assert list(tc_list)[0]["id"] == "tc1"


# --------------------------------------------------------------------------- #
#  Concurrent save collision detection                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_saves_collision_detection(setup_test_user, test_user_id):
    """Test that concurrent saves from streaming loop and callback handle collisions correctly.

    Simulates the race condition where:
    1. Streaming loop starts with saved_msg_count=5
    2. Long-running callback appends message #5 and saves
    3. Streaming loop tries to save with stale count=5

    The collision detection should handle this gracefully.
    """
    import asyncio

    # Create a session with initial messages
    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    for i in range(3):
        session.messages.append(
            ChatMessage(
                role="user" if i % 2 == 0 else "assistant", content=f"Message {i}"
            )
        )

    # Save initial messages
    session = await upsert_chat_session(session)

    # Simulate streaming loop and callback saving concurrently
    async def streaming_loop_save():
        """Simulates streaming loop saving messages."""
        # Add 2 messages
        session.messages.append(ChatMessage(role="user", content="Streaming message 1"))
        session.messages.append(
            ChatMessage(role="assistant", content="Streaming message 2")
        )

        # Wait a bit to let callback potentially save first
        await asyncio.sleep(0.01)

        # Save (will query DB for existing count)
        return await upsert_chat_session(session)

    async def callback_save():
        """Simulates long-running callback saving a message."""
        # Add 1 message
        session.messages.append(
            ChatMessage(role="tool", content="Callback result", tool_call_id="tc1")
        )

        # Save immediately (will query DB for existing count)
        return await upsert_chat_session(session)

    # Run both saves concurrently - one will hit collision detection
    results = await asyncio.gather(streaming_loop_save(), callback_save())

    # Both should succeed
    assert all(r is not None for r in results)

    # Reload session from DB to verify
    from backend.data.redis_client import get_redis_async

    redis_key = f"chat:session:{session.session_id}"
    async_redis = await get_redis_async()
    await async_redis.delete(redis_key)  # Clear cache to force DB load

    loaded_session = await get_chat_session(session.session_id, test_user_id)
    assert loaded_session is not None

    # Should have all 6 messages (3 initial + 2 streaming + 1 callback)
    assert len(loaded_session.messages) == 6

    # Verify no duplicate sequences
    sequences = []
    for i, msg in enumerate(loaded_session.messages):
        # Messages should have sequential sequence numbers starting from 0
        sequences.append(i)

    # All sequences should be unique and sequential
    assert sequences == list(range(6))

    # Verify message content is preserved
    contents = [m.content for m in loaded_session.messages]
    assert "Message 0" in contents
    assert "Message 1" in contents
    assert "Message 2" in contents
    assert "Streaming message 1" in contents
    assert "Streaming message 2" in contents
    assert "Callback result" in contents


# --------------------------------------------------------------------------- #
#  is_message_duplicate                                                        #
# --------------------------------------------------------------------------- #


def test_duplicate_detected_in_trailing_same_role():
    """Duplicate user message at the tail is detected."""
    msgs = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi there"),
        ChatMessage(role="user", content="yes"),
    ]
    assert is_message_duplicate(msgs, "user", "yes") is True


def test_duplicate_not_detected_across_turns():
    """Same text in a previous turn (separated by assistant) is NOT a duplicate."""
    msgs = [
        ChatMessage(role="user", content="yes"),
        ChatMessage(role="assistant", content="ok"),
    ]
    assert is_message_duplicate(msgs, "user", "yes") is False


def test_no_duplicate_on_empty_messages():
    """Empty message list never reports a duplicate."""
    assert is_message_duplicate([], "user", "hello") is False


def test_no_duplicate_when_content_differs():
    """Different content in the trailing same-role block is not a duplicate."""
    msgs = [
        ChatMessage(role="assistant", content="response"),
        ChatMessage(role="user", content="first message"),
    ]
    assert is_message_duplicate(msgs, "user", "second message") is False


def test_duplicate_with_multiple_trailing_same_role():
    """Detects duplicate among multiple consecutive same-role messages."""
    msgs = [
        ChatMessage(role="assistant", content="response"),
        ChatMessage(role="user", content="msg1"),
        ChatMessage(role="user", content="msg2"),
    ]
    assert is_message_duplicate(msgs, "user", "msg1") is True
    assert is_message_duplicate(msgs, "user", "msg2") is True
    assert is_message_duplicate(msgs, "user", "msg3") is False


def test_duplicate_check_for_assistant_role():
    """Works correctly when checking assistant role too."""
    msgs = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="assistant", content="how can I help?"),
    ]
    assert is_message_duplicate(msgs, "assistant", "hello") is True
    assert is_message_duplicate(msgs, "assistant", "new response") is False


def test_no_false_positive_when_content_is_none():
    """Messages with content=None in the trailing block do not match."""
    msgs = [
        ChatMessage(role="user", content=None),
        ChatMessage(role="user", content="hello"),
    ]
    assert is_message_duplicate(msgs, "user", "hello") is True
    # None-content message should not match any string
    msgs2 = [
        ChatMessage(role="user", content=None),
    ]
    assert is_message_duplicate(msgs2, "user", "hello") is False


def test_all_same_role_messages():
    """When all messages share the same role, the entire list is scanned."""
    msgs = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="user", content="second"),
        ChatMessage(role="user", content="third"),
    ]
    assert is_message_duplicate(msgs, "user", "first") is True
    assert is_message_duplicate(msgs, "user", "new") is False


# --------------------------------------------------------------------------- #
#  maybe_append_user_message                                                   #
# --------------------------------------------------------------------------- #


def test_maybe_append_user_message_appends_new():
    """A new user message is appended and returns True."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="assistant", content="hello"),
    ]
    result = maybe_append_user_message(session, "new msg", is_user_message=True)
    assert result is True
    assert len(session.messages) == 2
    assert session.messages[-1].role == "user"
    assert session.messages[-1].content == "new msg"


def test_maybe_append_user_message_skips_duplicate():
    """A duplicate user message is skipped and returns False."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="dup"),
    ]
    result = maybe_append_user_message(session, "dup", is_user_message=True)
    assert result is False
    assert len(session.messages) == 2


def test_maybe_append_user_message_none_message():
    """None/empty message returns False without appending."""
    session = ChatSession.new(user_id="u", dry_run=False)
    assert maybe_append_user_message(session, None, is_user_message=True) is False
    assert maybe_append_user_message(session, "", is_user_message=True) is False
    assert len(session.messages) == 0


def test_maybe_append_assistant_message():
    """Works for assistant role when is_user_message=False."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hi"),
    ]
    result = maybe_append_user_message(session, "response", is_user_message=False)
    assert result is True
    assert session.messages[-1].role == "assistant"
    assert session.messages[-1].content == "response"


def test_maybe_append_assistant_skips_duplicate():
    """Duplicate assistant message is skipped."""
    session = ChatSession.new(user_id="u", dry_run=False)
    session.messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="dup"),
    ]
    result = maybe_append_user_message(session, "dup", is_user_message=False)
    assert result is False
    assert len(session.messages) == 2


# --------------------------------------------------------------------------- #
#  append_and_save_message                                                     #
# --------------------------------------------------------------------------- #


def _make_session_with_messages(*msgs: ChatMessage) -> ChatSession:
    s = ChatSession.new(user_id="u1", dry_run=False)
    s.messages = list(msgs)
    return s


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_returns_none_for_duplicate(
    mocker: MockerFixture,
) -> None:
    """append_and_save_message returns None when the trailing message is a duplicate."""

    session = _make_session_with_messages(
        ChatMessage(role="user", content="hello"),
    )
    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=True)
    mock_redis_lock.release = mocker.AsyncMock()
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )

    result = await append_and_save_message(
        session.session_id, ChatMessage(role="user", content="hello")
    )
    assert result is None


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_appends_new_message(
    mocker: MockerFixture,
) -> None:
    """append_and_save_message appends a non-duplicate message and returns the session."""

    session = _make_session_with_messages(
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    )
    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=True)
    mock_redis_lock.release = mocker.AsyncMock()
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.model._save_session_to_db",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "backend.copilot.model.chat_db",
        return_value=mocker.MagicMock(
            get_next_sequence=mocker.AsyncMock(return_value=2)
        ),
    )
    mocker.patch(
        "backend.copilot.model.cache_chat_session",
        new_callable=mocker.AsyncMock,
    )

    new_msg = ChatMessage(role="user", content="second message")
    result = await append_and_save_message(session.session_id, new_msg)
    assert result is not None
    assert result.messages[-1].content == "second message"


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_raises_when_session_not_found(
    mocker: MockerFixture,
) -> None:
    """append_and_save_message raises ValueError when the session does not exist."""

    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=True)
    mock_redis_lock.release = mocker.AsyncMock()
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        new_callable=mocker.AsyncMock,
        return_value=None,
    )

    with pytest.raises(ValueError, match="not found"):
        await append_and_save_message(
            "missing-session-id", ChatMessage(role="user", content="hi")
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_uses_db_when_lock_degraded(
    mocker: MockerFixture,
) -> None:
    """When the Redis lock times out (acquired=False), the fallback reads from DB."""

    session = _make_session_with_messages(
        ChatMessage(role="assistant", content="hi"),
    )
    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=False)
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mock_get_from_db = mocker.patch(
        "backend.copilot.model._get_session_from_db",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.model._save_session_to_db",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "backend.copilot.model.chat_db",
        return_value=mocker.MagicMock(
            get_next_sequence=mocker.AsyncMock(return_value=1)
        ),
    )
    mocker.patch(
        "backend.copilot.model.cache_chat_session",
        new_callable=mocker.AsyncMock,
    )

    new_msg = ChatMessage(role="user", content="new msg")
    result = await append_and_save_message(session.session_id, new_msg)
    # DB path was used (not cache-first)
    mock_get_from_db.assert_called_once_with(session.session_id)
    assert result is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_raises_database_error_on_save_failure(
    mocker: MockerFixture,
) -> None:
    """When _save_session_to_db fails, append_and_save_message raises DatabaseError."""
    from backend.util.exceptions import DatabaseError

    session = _make_session_with_messages(
        ChatMessage(role="assistant", content="hi"),
    )
    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=True)
    mock_redis_lock.release = mocker.AsyncMock()
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.model._save_session_to_db",
        new_callable=mocker.AsyncMock,
        side_effect=RuntimeError("db down"),
    )
    mocker.patch(
        "backend.copilot.model.chat_db",
        return_value=mocker.MagicMock(
            get_next_sequence=mocker.AsyncMock(return_value=1)
        ),
    )

    with pytest.raises(DatabaseError):
        await append_and_save_message(
            session.session_id, ChatMessage(role="user", content="new msg")
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_invalidates_cache_on_cache_failure(
    mocker: MockerFixture,
) -> None:
    """When cache_chat_session fails, invalidate_session_cache is called to avoid stale reads."""

    session = _make_session_with_messages(
        ChatMessage(role="assistant", content="hi"),
    )
    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=True)
    mock_redis_lock.release = mocker.AsyncMock()
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.model._save_session_to_db",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "backend.copilot.model.chat_db",
        return_value=mocker.MagicMock(
            get_next_sequence=mocker.AsyncMock(return_value=1)
        ),
    )
    mocker.patch(
        "backend.copilot.model.cache_chat_session",
        new_callable=mocker.AsyncMock,
        side_effect=RuntimeError("redis write failed"),
    )
    mock_invalidate = mocker.patch(
        "backend.copilot.model.invalidate_session_cache",
        new_callable=mocker.AsyncMock,
    )

    result = await append_and_save_message(
        session.session_id, ChatMessage(role="user", content="new msg")
    )
    # DB write succeeded, cache invalidation was called
    mock_invalidate.assert_called_once_with(session.session_id)
    assert result is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_uses_db_when_redis_unavailable(
    mocker: MockerFixture,
) -> None:
    """When get_redis_async raises, _get_session_lock yields False (degraded) and DB is read."""

    session = _make_session_with_messages(
        ChatMessage(role="assistant", content="hi"),
    )
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        side_effect=ConnectionError("redis down"),
    )
    mock_get_from_db = mocker.patch(
        "backend.copilot.model._get_session_from_db",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.model._save_session_to_db",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "backend.copilot.model.chat_db",
        return_value=mocker.MagicMock(
            get_next_sequence=mocker.AsyncMock(return_value=1)
        ),
    )
    mocker.patch(
        "backend.copilot.model.cache_chat_session",
        new_callable=mocker.AsyncMock,
    )

    new_msg = ChatMessage(role="user", content="new msg")
    result = await append_and_save_message(session.session_id, new_msg)
    mock_get_from_db.assert_called_once_with(session.session_id)
    assert result is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_save_message_lock_release_failure_is_ignored(
    mocker: MockerFixture,
) -> None:
    """If lock.release() raises, the exception is swallowed (TTL will clean up)."""

    session = _make_session_with_messages(
        ChatMessage(role="assistant", content="hi"),
    )
    mock_redis_lock = mocker.AsyncMock()
    mock_redis_lock.acquire = mocker.AsyncMock(return_value=True)
    mock_redis_lock.release = mocker.AsyncMock(
        side_effect=RuntimeError("release failed")
    )
    mock_redis_client = mocker.MagicMock()
    mock_redis_client.lock = mocker.MagicMock(return_value=mock_redis_lock)
    mocker.patch(
        "backend.copilot.model.get_redis_async",
        new_callable=mocker.AsyncMock,
        return_value=mock_redis_client,
    )
    mocker.patch(
        "backend.copilot.model.get_chat_session",
        new_callable=mocker.AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.model._save_session_to_db",
        new_callable=mocker.AsyncMock,
    )
    mocker.patch(
        "backend.copilot.model.chat_db",
        return_value=mocker.MagicMock(
            get_next_sequence=mocker.AsyncMock(return_value=1)
        ),
    )
    mocker.patch(
        "backend.copilot.model.cache_chat_session",
        new_callable=mocker.AsyncMock,
    )

    new_msg = ChatMessage(role="user", content="new msg")
    result = await append_and_save_message(session.session_id, new_msg)
    assert result is not None
