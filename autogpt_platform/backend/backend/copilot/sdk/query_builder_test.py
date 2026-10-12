"""Tests for _format_conversation_context and _build_query_message."""

import logging
from datetime import UTC, datetime

import pytest

from backend.copilot.constants import COPILOT_ERROR_PREFIX, STOPPED_BY_USER_MARKER
from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionInfo
from backend.copilot.sdk.compaction import COMPACTION_TOOL_NAME, CompactionStats
from backend.copilot.sdk.service import (
    _BARE_MESSAGE_TOKEN_FLOOR,
    _build_query_message,
    _format_conversation_context,
    _transcript_watermark_after_turn,
)
from backend.copilot.sdk.turn_summary import TurnSummaryCollector

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
    # What the agent *did* is part of the history it restarts from; the
    # compressor budgets these arguments, so the renderer must show them.
    assert result is not None
    assert 'You called search: {"q": "test"}' in result


def test_format_names_autopilot_tools_as_the_sdk_declares_them():
    """History must not show the bare call notation the model copies into a
    call the CLI refuses: AutoPilot's own tools carry their declared
    ``mcp__copilot__`` name, CLI built-ins and other names stay as stored."""
    calls = [
        ("run_capability", '{"id": "tool:read_skill", "input": {"name": "x"}}'),
        ("glob", '{"pattern": "*.md"}'),
        ("TodoWrite", '{"todos": []}'),
        ("search", '{"q": "test"}'),
    ]
    msgs = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": f"tc{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
                for i, (name, arguments) in enumerate(calls)
            ],
        )
    ]

    result = _format_conversation_context(msgs)

    assert result is not None
    assert (
        'You called mcp__copilot__run_capability: {"id": "tool:read_skill", '
        '"input": {"name": "x"}}' in result
    )
    assert 'You called mcp__copilot__glob: {"pattern": "*.md"}' in result
    assert 'You called TodoWrite: {"todos": []}' in result
    assert 'You called search: {"q": "test"}' in result
    assert "You called run_capability" not in result


def test_format_tool_result_is_rendered_as_the_compressor_left_it():
    """No 500-character cut here: the compressor already sized this content,
    and cutting it again made the compression budget a fiction."""
    long_output = "x" * 5_000
    msgs = [ChatMessage(role="tool", content=long_output)]
    result = _format_conversation_context(msgs)
    assert result is not None
    assert f"Tool output: {long_output}" in result


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
    result, stats = await _build_query_message(
        "what's new?",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    # transcript_msg_count == msg_count - 1, so no gap
    assert result == "what's new?"
    assert stats is None


@pytest.mark.asyncio
async def test_build_query_resume_watermark_after_a_user_row_sends_the_rest():
    """The watermark is the next uncovered sequence, whatever role sits
    before it: rows at or above it are sent, rows below it are not."""
    session = _make_session(
        [
            ChatMessage(role="user", content="turn 1"),
            ChatMessage(role="assistant", content="reply 1"),
            ChatMessage(role="user", content="turn 2"),
            ChatMessage(role="assistant", content="reply 2"),
            ChatMessage(role="user", content="turn 3"),
        ]
    )
    result, stats = await _build_query_message(
        "turn 3",
        session,
        use_resume=True,
        transcript_msg_count=3,
        session_id="test-session",
    )
    assert result == (
        "<conversation_history>\nYou responded: reply 2\n</conversation_history>"
        "\n\nNow, the user says:\nturn 3"
    )
    assert stats is None


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
    result, stats = await _build_query_message(
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
    assert stats is None  # gap context does not compact


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
    result, stats = await _build_query_message(
        "new msg",
        session,
        use_resume=True,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert result == "new msg"
    assert stats is None


@pytest.mark.asyncio
async def test_build_query_no_resume_single_message():
    """Without --resume and only 1 message, return raw message."""
    session = _make_session([ChatMessage(role="user", content="first")])
    result, stats = await _build_query_message(
        "first",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert result == "first"
    assert stats is None


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
    async def _mock_compress(msgs, target_tokens=None):
        return msgs, False, None

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    result, stats = await _build_query_message(
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
    assert stats is None  # mock returns no stats


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
    result, stats = await _build_query_message(
        "current msg with pending1 pending2",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
        session_msg_ceiling=3,  # len(session.messages) before drain
    )
    # Gap should be empty (transcript_msg_count == ceiling - 1), so no history prepended
    assert result == "current msg with pending1 pending2"
    assert stats is None
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
    result, _stats = await _build_query_message(
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
async def test_build_query_session_msg_ceiling_suppresses_spurious_no_resume_fallback():
    """session_msg_ceiling prevents the no-resume compression fallback from
    firing on the first turn of a session when pending messages inflate msg_count.

    Scenario: fresh session (1 message) + 1 pending message drained at turn start.
    Without the ceiling: msg_count=2 > 1 → fallback triggers → pending message
    leaked into history → wrong context sent to model.
    With session_msg_ceiling=1 (pre-drain count): effective_count=1, 1 > 1 is False
    → fallback does not trigger → current_message returned as-is.
    """
    # session.messages after drain: [current_msg, pending_msg]
    session = _make_session(
        [
            ChatMessage(role="user", content="What is 2 plus 2?"),
            ChatMessage(role="user", content="What is 7 plus 7?"),  # pending
        ]
    )
    result, stats = await _build_query_message(
        "What is 2 plus 2?\n\nWhat is 7 plus 7?",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
        session_msg_ceiling=1,  # pre-drain: only 1 message existed
    )
    # Should return current_message directly without wrapping in history context
    assert result == "What is 2 plus 2?\n\nWhat is 7 plus 7?"
    assert stats is None
    # Pending question must NOT appear in a spurious history section
    assert "<conversation_history>" not in result


@pytest.mark.asyncio
async def test_build_query_no_resume_multi_message_compacted(monkeypatch):
    """When compression actually compacts, stats should be non-None."""
    session = _make_session(
        [
            ChatMessage(role="user", content="old"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="new"),
        ]
    )

    async def _mock_compress(msgs, target_tokens=None):
        # Simulate actual compaction
        return msgs, True, CompactionStats(messages_before=3, messages_after=3)

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    result, stats = await _build_query_message(
        "new",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
    )
    assert stats is not None


@pytest.mark.asyncio
async def test_build_query_no_resume_at_token_floor():
    """When target_tokens is at or below the floor, return bare message.

    This is the final escape hatch: if the retry budget is exhausted and
    even the most aggressive compression might not fit, skip history
    injection entirely so the user always gets a response.
    """
    session = _make_session(
        [
            ChatMessage(role="user", content="old question"),
            ChatMessage(role="assistant", content="old answer"),
            ChatMessage(role="user", content="new question"),
        ]
    )
    result, stats = await _build_query_message(
        "new question",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
        target_tokens=_BARE_MESSAGE_TOKEN_FLOOR,
    )
    # At the floor threshold, no history is injected
    assert result == "new question"
    assert stats is None


@pytest.mark.asyncio
async def test_build_query_no_resume_below_token_floor():
    """target_tokens strictly below floor also returns bare message."""
    session = _make_session(
        [
            ChatMessage(role="user", content="old"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="new"),
        ]
    )
    result, stats = await _build_query_message(
        "new",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
        target_tokens=_BARE_MESSAGE_TOKEN_FLOOR - 1,
    )
    assert result == "new"
    assert stats is None


@pytest.mark.asyncio
async def test_build_query_no_resume_above_token_floor_compresses(monkeypatch):
    """target_tokens just above the floor still triggers compression."""
    session = _make_session(
        [
            ChatMessage(role="user", content="old"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="new"),
        ]
    )

    async def _mock_compress(msgs, target_tokens=None):
        return msgs, False, None

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    result, _stats = await _build_query_message(
        "new",
        session,
        use_resume=False,
        transcript_msg_count=0,
        session_id="test-session",
        target_tokens=_BARE_MESSAGE_TOKEN_FLOOR + 1,
    )
    # Above the floor → history is injected (not the bare message)
    assert "<conversation_history>" in result
    assert "Now, the user says:\nnew" in result


# ---------------------------------------------------------------------------
# Cap-engaged: window starts above absolute sequence 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_query_resume_cap_engaged_uses_sequence_gap(monkeypatch):
    """When the eager-load cap engages, the windowed ``prior`` doesn't start
    at absolute sequence 0; index-based ``prior[transcript_msg_count:]`` is
    wrong.  ``transcript_msg_count`` IS the next uncovered DB sequence, so
    filtering by ``sequence >= watermark`` reads it directly."""
    # Window contains the 3 most-recent rows of a much longer conversation.
    # Watermark = 1500 (next uncovered DB sequence).
    session = _make_session(
        [
            ChatMessage(role="user", content="cap-window-user", sequence=1500),
            ChatMessage(role="assistant", content="cap-window-asst", sequence=1501),
            ChatMessage(role="user", content="current turn", sequence=1502),
        ]
    )

    async def _mock_compress(msgs, target_tokens=None):
        return msgs, False, None

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    # No DB lookup expected — the watermark is already a sequence.
    fake_db = type("FakeDB", (), {})()
    monkeypatch.setattr(
        "backend.copilot.sdk.resume_gap.chat_db",
        lambda: fake_db,
    )

    result, stats = await _build_query_message(
        "current turn",
        session,
        use_resume=True,
        transcript_msg_count=1500,
        session_id="test-session",
    )
    # Both windowed messages have sequence >= 1500 — they form the gap.
    assert "<conversation_history>" in result
    assert "cap-window-user" in result
    assert "cap-window-asst" in result
    assert "Now, the user says:\ncurrent turn" in result
    assert stats is None


@pytest.mark.asyncio
async def test_build_query_resume_cap_engaged_hole_fills_above_watermark(monkeypatch):
    """When the window starts above the watermark, the missing sequences are
    fetched from DB and prepended to ``window_gap`` so the LLM sees the full
    post-watermark conversation."""
    from datetime import UTC, datetime

    from backend.copilot.db import PaginatedMessages
    from backend.copilot.model import ChatSessionInfo

    # Window covers sequences 1502..1503; watermark sits at 1500 leaving
    # sequences 1500..1501 in the hole.
    session = _make_session(
        [
            ChatMessage(role="user", content="window-1502", sequence=1502),
            ChatMessage(role="assistant", content="window-1503", sequence=1503),
            ChatMessage(role="user", content="current turn", sequence=1504),
        ]
    )

    async def _mock_compress(msgs, target_tokens=None):
        return msgs, False, None

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    hole_page = PaginatedMessages(
        messages=[
            ChatMessage(role="user", content="hole-1500", sequence=1500),
            ChatMessage(role="assistant", content="hole-1501", sequence=1501),
        ],
        has_more=False,
        oldest_sequence=1500,
        session=ChatSessionInfo(
            session_id="test-session",
            user_id="u1",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
    )

    fake_db = type(
        "FakeDB",
        (),
        {
            "get_chat_messages_paginated": staticmethod(
                lambda *args, **kwargs: _async_return(hole_page)
            ),
        },
    )()
    monkeypatch.setattr(
        "backend.copilot.sdk.resume_gap.chat_db",
        lambda: fake_db,
    )

    result, _ = await _build_query_message(
        "current turn",
        session,
        use_resume=True,
        transcript_msg_count=1500,
        session_id="test-session",
    )
    # All 4 messages (2 hole + 2 window) are in the gap.
    assert "<conversation_history>" in result
    assert "hole-1500" in result
    assert "hole-1501" in result
    assert "window-1502" in result
    assert "window-1503" in result
    assert "Now, the user says:\ncurrent turn" in result


@pytest.mark.asyncio
async def test_build_query_resume_cap_engaged_hole_fill_failure_uses_window_only(
    monkeypatch,
):
    """If the hole-fill fetch raises, fall back to the windowed gap rather
    than dropping the entire LLM context."""
    session = _make_session(
        [
            ChatMessage(role="user", content="window-1502", sequence=1502),
            ChatMessage(role="assistant", content="window-1503", sequence=1503),
            ChatMessage(role="user", content="current", sequence=1504),
        ]
    )

    async def _mock_compress(msgs, target_tokens=None):
        return msgs, False, None

    monkeypatch.setattr(
        "backend.copilot.sdk.service._compress_messages",
        _mock_compress,
    )

    fake_db = type(
        "FakeDB",
        (),
        {
            "get_chat_messages_paginated": staticmethod(
                lambda *args, **kwargs: _async_raise(RuntimeError("db down"))
            ),
        },
    )()
    monkeypatch.setattr(
        "backend.copilot.sdk.resume_gap.chat_db",
        lambda: fake_db,
    )

    result, _ = await _build_query_message(
        "current",
        session,
        use_resume=True,
        transcript_msg_count=1500,
        session_id="test-session",
    )
    # Hole fetch failed but window_gap survives — context is partial but
    # not empty, and the turn keeps going.
    assert "<conversation_history>" in result
    assert "window-1502" in result
    assert "window-1503" in result
    assert "Now, the user says:\ncurrent" in result


@pytest.mark.asyncio
async def test_build_query_resume_hole_fill_drops_error_markers(monkeypatch):
    """Rows fetched from the DB for the hole go through the same marker
    filter as the loaded window."""
    session = _make_session(
        _seq_rows(
            ("user", "window-1502"),
            ("assistant", "window-1503"),
            ("user", "current turn"),
            start=1502,
        )
    )
    hole_page = PaginatedMessages(
        messages=_seq_rows(
            ("user", "hole-1500"),
            ("assistant", f"{COPILOT_ERROR_PREFIX} HOLE MARKER"),
            start=1500,
        ),
        has_more=False,
        oldest_sequence=1500,
        session=ChatSessionInfo(
            session_id="test-session",
            user_id="u1",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
    )
    fake_db = type(
        "FakeDB",
        (),
        {
            "get_chat_messages_paginated": staticmethod(
                lambda *args, **kwargs: _async_return(hole_page)
            )
        },
    )()
    monkeypatch.setattr("backend.copilot.sdk.resume_gap.chat_db", lambda: fake_db)

    result, _ = await _build_query_message(
        "current turn",
        session,
        use_resume=True,
        transcript_msg_count=1500,
        session_id="test-session",
    )

    assert "hole-1500" in result
    assert "window-1503" in result
    assert "HOLE MARKER" not in result


async def _async_return(value):
    return value


async def _async_raise(exc):
    raise exc


# ---------------------------------------------------------------------------
# Resumed-session gap: production-shaped rows (every row has a sequence)
# ---------------------------------------------------------------------------

_CAP_WARNING = "Cap-engaged + empty sequence-based gap"


def _seq_rows(*rows: tuple[str, str], start: int = 0) -> list[ChatMessage]:
    return [
        ChatMessage(role=role, content=content, sequence=start + i)
        for i, (role, content) in enumerate(rows)
    ]


def _history(*lines: str) -> str:
    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


def _cap_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and _CAP_WARNING in r.getMessage()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows, watermark, ceiling",
    [
        pytest.param(
            _seq_rows(("user", "q"), ("assistant", "a"), ("user", "CURRENT")),
            2,
            None,
            id="current-transcript",
        ),
        pytest.param(
            _seq_rows(
                ("user", "q"),
                ("reasoning", "r"),
                ("assistant", "a"),
                ("user", "CURRENT"),
            ),
            3,
            None,
            id="current-transcript-with-reasoning",
        ),
        pytest.param(
            _seq_rows(
                ("user", "q"),
                ("assistant", "a"),
                ("user", "CURRENT"),
                ("user", "pending drained at turn start"),
            ),
            2,
            3,
            id="turn-start-drain-after-ceiling",
        ),
        pytest.param(
            _seq_rows(
                ("user", "q"),
                ("assistant", "a"),
                ("reasoning", "r"),
                ("user", "CURRENT"),
            ),
            2,
            None,
            id="trailing-reasoning-row",
        ),
        pytest.param(
            _seq_rows(
                ("user", "q"), ("assistant", "a"), ("user", "CURRENT"), start=1000
            ),
            1002,
            None,
            id="capped-window-transcript-current",
        ),
    ],
)
async def test_build_query_resume_current_transcript_sends_no_gap_and_no_cap_warning(
    rows, watermark, ceiling, caplog
):
    """A healthy resumed turn whose transcript covers every prior row sends
    the bare message and does not log the cap warning (it used to fire on
    847 of 886 resumed prod turns, all of them current)."""
    result, stats = await _build_query_message(
        "CURRENT",
        _make_session(rows),
        use_resume=True,
        transcript_msg_count=watermark,
        session_id="test-session",
        session_msg_ceiling=ceiling,
    )
    assert result == "CURRENT"
    assert stats is None
    assert _cap_warnings(caplog) == []


@pytest.mark.asyncio
async def test_build_query_resume_cap_warning_fires_when_unloaded_rows_are_left_out(
    monkeypatch, caplog
):
    """The loaded window starts above the watermark, the hole fetch fails and
    the window holds nothing sendable: rows were left out, so it warns, with
    its text and field order unchanged."""
    session = _make_session(
        _seq_rows(
            ("assistant", f"{COPILOT_ERROR_PREFIX} Operation cancelled"),
            ("user", "CURRENT"),
            start=1502,
        )
    )
    fake_db = type(
        "FakeDB",
        (),
        {
            "get_chat_messages_paginated": staticmethod(
                lambda *args, **kwargs: _async_raise(RuntimeError("db down"))
            ),
        },
    )()
    monkeypatch.setattr("backend.copilot.sdk.resume_gap.chat_db", lambda: fake_db)

    result, _ = await _build_query_message(
        "CURRENT",
        session,
        use_resume=True,
        transcript_msg_count=1500,
        session_id="test-session",
    )
    assert result == "CURRENT"
    assert _cap_warnings(caplog) == [
        "[SDK] [test-ses] Cap-engaged + empty sequence-based gap: window may not"
        " contain post-watermark rows (transcript=1500, db=2)"
    ]


@pytest.mark.asyncio
async def test_build_query_resume_cap_warning_fires_when_only_the_window_is_sent(
    monkeypatch, caplog
):
    """The hole fetch fails but the loaded rows above the watermark are still
    sent: the rows in between were left out, so it warns as well."""
    session = _make_session(
        _seq_rows(
            ("user", "LOADED QUESTION"),
            ("assistant", "LOADED ANSWER"),
            ("user", "CURRENT"),
            start=1504,
        )
    )
    fake_db = type(
        "FakeDB",
        (),
        {
            "get_chat_messages_paginated": staticmethod(
                lambda *args, **kwargs: _async_raise(RuntimeError("db down"))
            ),
        },
    )()
    monkeypatch.setattr("backend.copilot.sdk.resume_gap.chat_db", lambda: fake_db)

    result, _ = await _build_query_message(
        "CURRENT",
        session,
        use_resume=True,
        transcript_msg_count=1500,
        session_id="test-session",
    )
    assert result == (
        _history("User: LOADED QUESTION", "You responded: LOADED ANSWER")
        + "\n\nNow, the user says:\nCURRENT"
    )
    assert _cap_warnings(caplog) == [
        "[SDK] [test-ses] Cap-engaged + empty sequence-based gap: window may not"
        " contain post-watermark rows (transcript=1500, db=3)"
    ]


def test_transcript_watermark_after_a_midturn_followup_covers_the_whole_turn():
    """The follow-up row is not subtracted: the watermark is one past the
    turn's last row, so nothing of the turn is uncovered."""
    turn = _seq_rows(
        ("user", "build me an agent"),
        ("assistant", "On it."),
        ("tool", "TOOL OUTPUT"),
        ("user", "FOLLOW-UP: make it hourly"),
        ("reasoning", "thinking again"),
        ("assistant", "FINAL ANSWER"),
        start=40,
    )
    assert _transcript_watermark_after_turn(turn) == 46


@pytest.mark.asyncio
async def test_build_query_after_a_midturn_followup_turn_resends_nothing():
    """Turn N took a follow-up mid-turn.  Its text reached the CLI as the
    PostToolUse hook's additionalContext, which the CLI stores in its session
    file (see followup_resume_cli_test.py), so the watermark uploaded after
    turn N covers every row of the turn.  Turn N+1 must not re-send turn N's
    tail: a watermark stepped back by the follow-up count sends the last tool
    output and the final answer as a <conversation_history> prefix, and not
    the follow-up."""
    turn_n = _seq_rows(
        ("user", "build me an agent"),
        ("reasoning", "thinking"),
        ("assistant", "On it."),
        ("tool", "TOOL OUTPUT"),
        ("user", "FOLLOW-UP: make it hourly"),
        ("reasoning", "thinking again"),
        ("assistant", "FINAL ANSWER"),
    )
    watermark = _transcript_watermark_after_turn(turn_n)
    session = _make_session(turn_n + _seq_rows(("user", "CURRENT"), start=7))

    result, stats = await _build_query_message(
        "CURRENT",
        session,
        use_resume=True,
        transcript_msg_count=watermark,
        session_id="test-session",
    )
    assert watermark == 7
    assert result == "CURRENT"
    assert stats is None


@pytest.mark.asyncio
async def test_build_query_resume_gap_never_sends_error_markers():
    """An error marker above the watermark is UI state, not something the
    model said: it must not reach the model as 'You responded: ...'."""
    session = _make_session(
        _seq_rows(
            ("user", "q"),
            ("assistant", "a"),
            ("user", "MY QUESTION"),
            ("assistant", f"{COPILOT_ERROR_PREFIX} Operation cancelled"),
            ("assistant", STOPPED_BY_USER_MARKER),
            ("user", "CURRENT"),
        )
    )
    result, _ = await _build_query_message(
        "CURRENT",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    assert result == (
        _history("User: MY QUESTION") + "\n\nNow, the user says:\nCURRENT"
    )


@pytest.mark.asyncio
async def test_build_query_resume_gap_of_only_a_marker_sends_the_bare_message(
    caplog,
):
    session = _make_session(
        _seq_rows(
            ("user", "q"),
            ("assistant", "a"),
            ("assistant", f"{COPILOT_ERROR_PREFIX} Operation cancelled"),
            ("user", "CURRENT"),
        )
    )
    result, _ = await _build_query_message(
        "CURRENT",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    assert result == "CURRENT"
    assert _cap_warnings(caplog) == []


@pytest.mark.asyncio
async def test_build_query_resume_gap_hands_no_compaction_rows_to_the_compressor(
    monkeypatch,
):
    compressed_inputs: list[list[ChatMessage]] = []

    async def _spy(msgs, target_tokens=None):
        compressed_inputs.append(list(msgs))
        return msgs, False, None

    monkeypatch.setattr("backend.copilot.sdk.service._compress_messages", _spy)
    rows = _seq_rows(
        ("user", "q"),
        ("assistant", "a"),
        ("user", "Q"),
        ("assistant", ""),
        ("tool", "Conversation compacted"),
        ("assistant", "ANSWER"),
        ("user", "CURRENT"),
    )
    rows[3].tool_calls = [
        {
            "id": "compact-1",
            "type": "function",
            "function": {"name": COMPACTION_TOOL_NAME, "arguments": "{}"},
        }
    ]
    rows[4].tool_call_id = "compact-1"

    result, _ = await _build_query_message(
        "CURRENT",
        _make_session(rows),
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
    )
    assert [[m.content for m in batch] for batch in compressed_inputs] == [
        ["Q", "ANSWER"]
    ]
    assert result == (
        _history("User: Q", "You responded: ANSWER")
        + "\n\nNow, the user says:\nCURRENT"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows, watermark, expected_lines",
    [
        pytest.param(
            _seq_rows(
                ("user", "q"),
                ("reasoning", "r"),
                ("assistant", "a"),
                ("user", "FAILED USER"),
                ("reasoning", "r"),
                ("assistant", "FAILED PARTIAL"),
                ("user", "CURRENT"),
            ),
            3,
            ["User: FAILED USER", "You responded: FAILED PARTIAL"],
            id="failed-turn-was-skipped-as-misaligned",
        ),
        pytest.param(
            _seq_rows(
                ("user", "q"),
                ("reasoning", "r"),
                ("assistant", "a"),
                ("user", "q2"),
                ("reasoning", "r"),
                ("assistant", "a2"),
                ("user", "F1 USER"),
                ("assistant", "F1 PARTIAL"),
                ("user", "F2 USER"),
                ("assistant", "F2 PARTIAL"),
                ("user", "F3 USER"),
                ("assistant", "F3 PARTIAL"),
                ("user", "CURRENT"),
            ),
            6,
            [
                "User: F1 USER",
                "You responded: F1 PARTIAL",
                "User: F2 USER",
                "You responded: F2 PARTIAL",
                "User: F3 USER",
                "You responded: F3 PARTIAL",
            ],
            id="first-failed-turn-was-dropped",
        ),
        pytest.param(
            _seq_rows(
                ("user", "q"),
                ("reasoning", "r"),
                ("assistant", "a"),
                ("user", "q2"),
                ("reasoning", "r"),
                ("assistant", "a2"),
                ("user", "FAILED USER"),
                ("reasoning", "r"),
                ("assistant", "FAILED TEXT"),
                ("tool", "TOOL OUT"),
                ("user", "CURRENT"),
            ),
            6,
            [
                "User: FAILED USER",
                "You responded: FAILED TEXT",
                "Tool output: TOOL OUT",
            ],
            id="user-and-assistant-were-dropped-for-an-orphan-tool-output",
        ),
    ],
)
async def test_build_query_resume_gap_is_sequence_based_with_reasoning_rows(
    rows, watermark, expected_lines
):
    """Reasoning rows take sequences but never reach the CLI.  The gap is the
    rows at or above the watermark sequence, never a list position in a
    reasoning-filtered list."""
    result, _ = await _build_query_message(
        "CURRENT",
        _make_session(rows),
        use_resume=True,
        transcript_msg_count=watermark,
        session_id="test-session",
    )
    assert result == _history(*expected_lines) + "\n\nNow, the user says:\nCURRENT"


@pytest.mark.asyncio
async def test_build_query_resume_gap_rows_reach_the_turn_summary():
    """The rows sent as gap on a resumed turn are the history rows the turn's
    summary line reports."""
    session = _make_session(
        _seq_rows(
            ("user", "turn 1"),
            ("assistant", "reply 1"),
            ("user", "turn 2"),
            ("assistant", "reply 2"),
            ("user", "turn 3"),
        )
    )
    collector = TurnSummaryCollector("test-session")

    query, _ = await _build_query_message(
        "turn 3",
        session,
        use_resume=True,
        transcript_msg_count=2,
        session_id="test-session",
        turn_summary=collector,
    )
    collector.record_query(query, use_resume=True, covered_rows=2, session_rows=5)

    assert collector.summary.query_path == "resume_gap"
    assert collector.summary.history_rows == 2
