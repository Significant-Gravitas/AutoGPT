"""Tests for the resumed-session gap helpers."""

from unittest.mock import AsyncMock, MagicMock

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    STOPPED_BY_USER_MARKER,
)
from backend.copilot.model import ChatMessage
from backend.copilot.sdk.compaction import COMPACTION_TOOL_NAME
from backend.copilot.sdk.resume_gap import (
    is_marker_row,
    missing_range,
    sendable_gap,
    uncovered_rows,
)


def _rows(*roles: str, start: int = 0) -> list[ChatMessage]:
    return [
        ChatMessage(role=role, content=f"{role}-{start + i}", sequence=start + i)
        for i, role in enumerate(roles)
    ]


def test_uncovered_rows_reads_the_watermark_as_a_sequence_with_reasoning_rows():
    # 0:u 1:r 2:a | 3:u 4:r 5:a — the watermark 3 sits after the first turn.
    prior = _rows("user", "reasoning", "assistant", "user", "reasoning", "assistant")
    assert [m.content for m in uncovered_rows(prior, 3)] == [
        "user-3",
        "assistant-5",
    ]


def test_uncovered_rows_uses_list_position_for_rows_without_a_sequence():
    prior = [
        ChatMessage(role="user", content="u0"),
        ChatMessage(role="reasoning", content="r1"),
        ChatMessage(role="assistant", content="a2"),
        ChatMessage(role="user", content="u3"),
    ]
    assert [m.content for m in uncovered_rows(prior, 2)] == ["a2", "u3"]


def test_uncovered_rows_empty_when_the_transcript_is_current():
    assert uncovered_rows(_rows("user", "assistant"), 2) == []


def test_missing_range_only_when_the_window_starts_above_the_watermark():
    window = _rows("user", "assistant", start=1502)
    assert missing_range(window, 1500) == (1500, 1502)
    assert missing_range(window, 1502) is None
    assert missing_range(window, 1700) is None
    assert missing_range([], 5) is None


def test_rows_without_a_sequence_continue_from_the_row_before_them():
    # A capped window (1000, 1001) followed by two rows a failed save left
    # without a sequence: they sit at 1002 and 1003, not at list positions 2, 3.
    window = _rows("user", "assistant", start=1000) + [
        ChatMessage(role="user", content="unsaved-question"),
        ChatMessage(role="assistant", content="unsaved-answer"),
    ]
    assert missing_range(window, 900) == (900, 1000)
    assert [m.content for m in uncovered_rows(window, 1002)] == [
        "unsaved-question",
        "unsaved-answer",
    ]
    assert uncovered_rows(window, 1004) == []


def test_missing_range_is_unknown_when_no_row_has_a_sequence():
    assert missing_range([ChatMessage(role="user", content="u")], 5) is None


def test_missing_range_counts_a_leading_reasoning_row_as_loaded():
    window = [ChatMessage(role="reasoning", content="r", sequence=10)] + _rows(
        "assistant", start=11
    )
    assert missing_range(window, 10) is None


def test_is_marker_row_matches_every_ui_marker_prefix():
    for content in (
        f"{COPILOT_ERROR_PREFIX} Operation cancelled",
        f"{COPILOT_RETRYABLE_ERROR_PREFIX} Try again",
        STOPPED_BY_USER_MARKER,
    ):
        assert is_marker_row(ChatMessage(role="assistant", content=content))
    assert not is_marker_row(ChatMessage(role="assistant", content="a real answer"))
    assert not is_marker_row(
        ChatMessage(role="user", content=f"{COPILOT_ERROR_PREFIX} pasted by a user")
    )
    assert not is_marker_row(ChatMessage(role="assistant", content=None))


def test_sendable_gap_drops_markers_and_compaction_rows():
    compaction_call = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "compact-1",
                "type": "function",
                "function": {"name": COMPACTION_TOOL_NAME, "arguments": "{}"},
            }
        ],
    )
    compaction_result = ChatMessage(
        role="tool", content="Conversation compacted", tool_call_id="compact-1"
    )
    rows = [
        ChatMessage(role="user", content="question"),
        ChatMessage(role="assistant", content=f"{COPILOT_ERROR_PREFIX} cancelled"),
        compaction_call,
        compaction_result,
        ChatMessage(role="assistant", content="answer"),
    ]
    assert [m.content for m in sendable_gap(rows)] == ["question", "answer"]


def _db_returning(rows: list[ChatMessage]) -> MagicMock:
    db = MagicMock()
    db.get_chat_messages_paginated = AsyncMock(return_value=MagicMock(messages=rows))
    return db
