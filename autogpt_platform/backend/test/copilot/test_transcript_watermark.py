"""Unit tests for the transcript watermark (message_count) fix.

The bug: upload used message_count=len(session.messages) (DB count).  When a
prior turn's GCS upload failed silently, the JSONL on GCS was stale (e.g.
covered only T1-T12) but the meta.json watermark matched the full DB count
(e.g. 46).  The next turn's gap-fill check (transcript_msg_count < msg_count-1)
never triggered, so the model silently lost context for the skipped turns.

The fix: watermark = previous_coverage + 2 (current user+asst pair) when
use_resume=True and transcript_msg_count > 0.  This ensures the watermark
reflects the JSONL content, not the DB count.

These tests exercise _build_query_message directly to verify that gap-fill
triggers with the corrected watermark but NOT with the inflated (buggy) one.
"""

from unittest.mock import MagicMock

import pytest

from backend.copilot.sdk.service import _build_query_message


def _make_messages(n_pairs: int, *, current_user: str = "current") -> list[MagicMock]:
    """Build a flat list of n_pairs*2 alternating user/asst messages, plus
    one trailing user message for the *current* turn.

    ``sequence`` is left ``None`` so the cap-engaged sequence-based path
    in ``_build_query_message`` (which short-circuits on ``sequence > 0``)
    doesn't fire — these tests target the legacy index-based gap branch.
    """
    msgs: list[MagicMock] = []
    for i in range(n_pairs):
        u = MagicMock()
        u.role = "user"
        u.content = f"user message {i}"
        u.sequence = None
        a = MagicMock()
        a.role = "assistant"
        a.content = f"assistant response {i}"
        a.sequence = None
        msgs.extend([u, a])
    # Current turn's user message
    cur = MagicMock()
    cur.role = "user"
    cur.content = current_user
    cur.sequence = None
    msgs.append(cur)
    return msgs


def _make_session(messages: list[MagicMock]) -> MagicMock:
    session = MagicMock()
    session.messages = messages
    return session


@pytest.mark.asyncio
async def test_gap_fill_triggers_for_stale_jsonl():
    """Scenario: T1-T12 in JSONL (watermark=24), DB has T1-T22+Test (46 msgs).

    With the FIX: 'Test' uploaded watermark=26 (T12's 24 + 2 for 'Test').
    Next turn (T24) downloads watermark=26, DB has 47.
    Gap check: 26 < 47-1=46 → TRUE → gap fills T14-T23.
    """
    # T23 turns in DB (46 messages) + T24 user = 47
    msgs = _make_messages(23, current_user="memory test - recall all")
    assert len(msgs) == 47

    session = _make_session(msgs)

    # Watermark as uploaded by the FIX: T12 covered 24, 'Test' +2 = 26
    result_msg, _ = await _build_query_message(
        current_message="memory test - recall all",
        session=session,
        use_resume=True,
        transcript_msg_count=26,
        session_id="test-session-id",
    )

    assert "<conversation_history>" in result_msg, (
        "Expected gap-fill to inject <conversation_history> when "
        "watermark=26 < msg_count-1=46"
    )


@pytest.mark.asyncio
async def test_no_gap_fill_when_watermark_is_current():
    """When the JSONL is fully current (watermark = DB-1), no gap injected."""
    # T23 turns in DB (46 messages) + T24 user = 47
    msgs = _make_messages(23, current_user="next message")
    session = _make_session(msgs)

    result_msg, _ = await _build_query_message(
        current_message="next message",
        session=session,
        use_resume=True,
        transcript_msg_count=46,  # current — no gap
        session_id="test-session-id",
    )

    assert (
        "<conversation_history>" not in result_msg
    ), "No gap-fill expected when watermark is current"
    assert result_msg == "next message"


@pytest.mark.asyncio
async def test_inflated_watermark_suppresses_gap_fill():
    """Documents the original bug: inflated watermark suppresses gap-fill.

    'Test' uploaded watermark=len(session.messages)=46 even though only 26
    messages are in the JSONL.  Next turn: 46 < 47-1=46 → FALSE → no gap fill.
    """
    msgs = _make_messages(23, current_user="memory test")
    session = _make_session(msgs)

    # Buggy watermark: inflated to DB count
    result_msg, _ = await _build_query_message(
        current_message="memory test",
        session=session,
        use_resume=True,
        transcript_msg_count=46,  # inflated — suppresses gap fill
        session_id="test-session-id",
    )

    assert (
        "<conversation_history>" not in result_msg
    ), "With inflated watermark, gap-fill is suppressed — this documents the bug"


@pytest.mark.asyncio
async def test_fixed_watermark_fills_same_gap():
    """Same scenario but with the FIXED watermark triggers gap-fill."""
    msgs = _make_messages(23, current_user="memory test")
    session = _make_session(msgs)

    result_msg, _ = await _build_query_message(
        current_message="memory test",
        session=session,
        use_resume=True,
        transcript_msg_count=26,  # fixed watermark
        session_id="test-session-id",
    )

    assert (
        "<conversation_history>" in result_msg
    ), "With fixed watermark=26, gap-fill triggers and injects missing turns"
