"""Tests for _format_conversation_context and _build_query_message."""

from datetime import UTC, datetime

import pytest

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import (
    _build_query_message,
    _format_conversation_context,
)

# ---------------------------------------------------------------------------
# _format_conversation_context
# ---------------------------------------------------------------------------


def test_format_empty_list():
    assert _format_conversation_context([]) is None


def test_format_none_content_messages():
    msgs = [ChatMessage(role="user", content=None)]
    assert _format_conversation_context(msgs) is None


def test_format_user_message():
    msgs = [ChatMessage(role="user", content="hello")]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert "User: hello" in result
    assert result.startswith("<conversation_history>")
    assert result.endswith("</conversation_history>")


def test_format_assistant_text():
    msgs = [ChatMessage(role="assistant", content="hi there")]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert "You responded: hi there" in result


def test_format_assistant_tool_calls():
    msgs = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"function": {"name": "search", "arguments": '{"q": "test"}'}}],
        )
    ]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert 'You called tool: search({"q": "test"})' in result


def test_format_tool_result():
    msgs = [ChatMessage(role="tool", content='{"result": "ok"}')]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert 'Tool result: {"result": "ok"}' in result


def test_format_tool_result_none_content():
    msgs = [ChatMessage(role="tool", content=None)]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert "Tool result: " in result


def test_format_full_conversation():
    msgs = [
        ChatMessage(role="user", content="find agents"),
        ChatMessage(
            role="assistant",
            content="I'll search for agents.",
            tool_calls=[
                {"function": {"name": "find_agents", "arguments": '{"q": "test"}'}}
            ],
        ),
        ChatMessage(role="tool", content='[{"id": "1", "name": "Agent1"}]'),
        ChatMessage(role="assistant", content="Found Agent1."),
    ]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert "User: find agents" in result
    assert "You responded: I'll search for agents." in result
    assert "You called tool: find_agents" in result
    assert "Tool result:" in result
    assert "You responded: Found Agent1." in result


# ---------------------------------------------------------------------------
# _build_query_message
# ---------------------------------------------------------------------------


def _make_session(messages: list[ChatMessage]) -> ChatSession:
    """Build a minimal ChatSession with the given messages."""
    now = datetime.now(UTC)
    return ChatSession(
        session_id="test-session",
        user_id="user-1",
        messages=messages,
        title="test",
        usage=[],
        started_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_build_query_resume_up_to_date():
    """With --resume and transcript covers all messages, return raw message."""
    session = _make_session(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="what's new?"),
        ]
    )
    result = await _build_query_message(
        "what's new?",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    # transcript_msg_count == msg_count - 1, so no gap
    assert result == "what's new?"


@pytest.mark.asyncio
async def test_build_query_resume_stale_transcript():
    """With --resume and stale transcript, gap context is prepended."""
    session = _make_session(
        [
            ChatMessage(role="user", content="turn 1"),
            ChatMessage(role="assistant", content="reply 1"),
            ChatMessage(role="user", content="turn 2"),
            ChatMessage(role="assistant", content="reply 2"),
            ChatMessage(role="user", content="turn 3"),
        ]
    )
    result = await _build_query_message(
        "turn 3",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    assert "<conversation_history>" in result
    assert "turn 2" in result
    assert "reply 2" in result
    assert "Now, the user says:\nturn 3" in result


@pytest.mark.asyncio
async def test_build_query_resume_zero_msg_count():
    """With --resume but transcript_msg_count=0, return raw message."""
    session = _make_session(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="new msg"),
        ]
    )
    result = await _build_query_message(
        "new msg",
        session,
        use_resume=True,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert result == "new msg"


@pytest.mark.asyncio
async def test_build_query_no_resume_single_message():
    """Without --resume and only 1 message, return raw message."""
    session = _make_session([ChatMessage(role="user", content="first")])
    result = await _build_query_message(
        "first",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert result == "first"


@pytest.mark.asyncio
async def test_build_query_no_resume_multi_message(monkeypatch):
    """Without --resume and multiple messages, compress and prepend."""
    session = _make_session(
        [
            ChatMessage(role="user", content="older question"),
            ChatMessage(role="assistant", content="older answer"),
            ChatMessage(role="user", content="new question"),
        ]
    )

    # Mock _compress_conversation_history to return the messages as-is
    async def _mock_compress(sess):
        return sess.messages[:-1]

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_conversation_history",
        _mock_compress,
    )

    result = await _build_query_message(
        "new question",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert "<conversation_history>" in result
    assert "older question" in result
    assert "older answer" in result
    assert "Now, the user says:\nnew question" in result
