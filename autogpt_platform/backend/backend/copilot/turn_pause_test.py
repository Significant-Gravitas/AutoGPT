from datetime import UTC, datetime

import pytest

from backend.copilot.constants import COPILOT_ERROR_PREFIX, COPILOT_SYSTEM_PREFIX
from backend.copilot.continuation import KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.turn_pause import (
    PAUSE_KEY,
    TurnPause,
    append_pause_marker,
    pause_of,
)


def _session(*messages: ChatMessage) -> ChatSession:
    now = datetime(2026, 10, 14, tzinfo=UTC)
    return ChatSession(
        session_id="s1",
        user_id="u1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=list(messages),
    )


def test_marker_row_carries_the_pause_and_a_readable_fallback():
    session = _session(ChatMessage(role="assistant", content="Where I got to: ..."))
    pause = TurnPause(reason="weekly", resets_at=datetime(2026, 10, 19, tzinfo=UTC))
    assert append_pause_marker(session, pause) is True
    row = session.messages[-1]
    assert row.role == "assistant"
    assert row.content is not None
    assert row.content.startswith(COPILOT_SYSTEM_PREFIX)
    assert "this week's budget" in row.content
    assert pause_of(row) == pause


def test_one_pause_per_turn():
    session = _session(ChatMessage(role="assistant", content="done so far"))
    pause = TurnPause(reason="steps", steps=450)
    assert append_pause_marker(session, pause) is True
    assert append_pause_marker(session, pause) is False
    assert len(session.messages) == 2


def test_no_pause_on_top_of_an_error_marker():
    session = _session(
        ChatMessage(role="assistant", content=f"{COPILOT_ERROR_PREFIX} broke")
    )
    assert append_pause_marker(session, TurnPause(reason="task")) is False


def test_rows_without_a_valid_pause_read_as_none():
    assert pause_of(ChatMessage(role="assistant", content="hello")) is None
    assert pause_of(ChatMessage(role="user", content="x", metadata={})) is None
    broken = ChatMessage(
        role="assistant",
        content=f"{COPILOT_SYSTEM_PREFIX} Paused",
        metadata={PAUSE_KEY: {"reason": "not-a-reason"}},
    )
    assert pause_of(broken) is None


@pytest.mark.parametrize("note", [KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE])
def test_a_kickoff_note_is_not_a_marker(note):
    # A continuation that pauses before writing anything still gets its
    # pause row: the note it started from begins a turn, it doesn't end one.
    session = _session(ChatMessage(role="assistant", content=note))
    assert append_pause_marker(session, TurnPause(reason="daily")) is True
    assert len(session.messages) == 2
