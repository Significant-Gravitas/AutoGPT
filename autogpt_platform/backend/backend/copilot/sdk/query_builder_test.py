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
    # Assistant with no content and tool_calls omitted produces no lines
    assert result is None


def test_format_tool_result():
    msgs = [ChatMessage(role="tool", content='{"result": "ok"}')]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert 'Tool output: {"result": "ok"}' in result


def test_format_tool_result_none_content():
    msgs = [ChatMessage(role="tool", content=None)]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert "Tool output: " in result


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
    # tool_calls are omitted to prevent model mimicry
    assert "Tool output:" in result
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
    result, was_compacted = await _build_query_message(
        "what's new?",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    # transcript_msg_count == msg_count - 1, so no gap
    assert result == "what's new?"
    assert was_compacted is False


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
    result, was_compacted = await _build_query_message(
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
    assert was_compacted is False  # gap context does not compact


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
    result, was_compacted = await _build_query_message(
        "new msg",
        session,
        use_resume=True,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert result == "new msg"
    assert was_compacted is False


@pytest.mark.asyncio
async def test_build_query_no_resume_single_message():
    """Without --resume and only 1 message, return raw message."""
    session = _make_session([ChatMessage(role="user", content="first")])
    result, was_compacted = await _build_query_message(
        "first",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert result == "first"
    assert was_compacted is False


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

    # Mock _compress_messages to return the messages as-is
    async def _mock_compress(msgs):
        return msgs, False

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    result, was_compacted = await _build_query_message(
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
    assert was_compacted is False  # mock returns False


@pytest.mark.asyncio
async def test_build_query_session_msg_ceiling_prevents_pending_duplication():
    """session_msg_ceiling stops pending messages from leaking into the gap.

    Scenario: transcript covers 2 messages, session has 2 historical + 1 current
    + 2 pending drained at turn start.  Without the ceiling the gap would include
    the pending messages AND current_message already has them → duplication.
    With session_msg_ceiling=3 (pre-drain count) the gap slice is empty and
    only current_message carries the pending content.
    """
    # session.messages after drain: [hist1, hist2, current_msg, pending1, pending2]
    session = _make_session(
        [
            ChatMessage(role="user", content="hist1"),
            ChatMessage(role="assistant", content="hist2"),
            ChatMessage(role="user", content="current msg with pending1 pending2"),
            ChatMessage(role="user", content="pending1"),
            ChatMessage(role="user", content="pending2"),
        ]
    )
    # transcript covers hist1+hist2 (2 messages); pre-drain count was 3 (includes current_msg)
    result, was_compacted = await _build_query_message(
        "current msg with pending1 pending2",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
        session_msg_ceiling=3,  # len(session.messages) before drain
    )
    # Gap should be empty (transcript_msg_count == ceiling - 1), so no history prepended
    assert result == "current msg with pending1 pending2"
    assert was_compacted is False
    # Pending messages must NOT appear in gap context
    assert "pending1" not in result.split("current msg")[0]


@pytest.mark.asyncio
async def test_build_query_session_msg_ceiling_preserves_real_gap():
    """session_msg_ceiling still surfaces a genuine stale-transcript gap.

    Scenario: transcript covers 2 messages, session has 4 historical + 1 current
    + 2 pending.  Ceiling = 5 (pre-drain).  Real gap = messages 2-3 (hist3, hist4).
    """
    session = _make_session(
        [
            ChatMessage(role="user", content="hist1"),
            ChatMessage(role="assistant", content="hist2"),
            ChatMessage(role="user", content="hist3"),
            ChatMessage(role="assistant", content="hist4"),
            ChatMessage(role="user", content="current"),
            ChatMessage(role="user", content="pending1"),
            ChatMessage(role="user", content="pending2"),
        ]
    )
    result, was_compacted = await _build_query_message(
        "current",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
        session_msg_ceiling=5,  # pre-drain: [hist1..hist4, current]
    )
    # Gap = session.messages[2:4] = [hist3, hist4]
    assert "<conversation_history>" in result
    assert "hist3" in result
    assert "hist4" in result
    assert "Now, the user says:\ncurrent" in result
    # Pending messages must NOT appear in gap
    assert "pending1" not in result
    assert "pending2" not in result


@pytest.mark.asyncio
async def test_build_query_no_resume_multi_message_compacted(monkeypatch):
    """When compression actually compacts, was_compacted should be True."""
    session = _make_session(
        [
            ChatMessage(role="user", content="old"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="new"),
        ]
    )

    async def _mock_compress(msgs):
        return msgs, True  # Simulate actual compaction

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    result, was_compacted = await _build_query_message(
        "new",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert was_compacted is True
