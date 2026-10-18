"""The first SDK turn on a chat with no CLI session to resume.

A local chat from before the one-engine upgrade (its turns ran on the old fast
path, whose transcripts the SDK does not resume), or any chat whose CLI session
file is gone, has stored messages but no ``--resume`` source. The first SDK
turn must inject the full prior session, not just a "gap" since a transcript's
end, because the CLI has zero context without ``--resume``. That turn sets
``session_id``, so the CLI writes its session file and later turns resume it.

Scenario table (SDK ``_build_query_message``)
=============================================

| # | Scenario                         | use_resume | tmc | Expected query message          |
|---|----------------------------------|------------|-----|---------------------------------|
| P | first SDK turn, 2 earlier turns  | False      | 4   | full session injected           |
| Q | the turn after it                | True       | 6   | bare message only (--resume ok) |
| R | first SDK turn, 1 earlier turn   | False      | 2   | full session injected           |
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import _build_query_message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(messages: list[ChatMessage]) -> ChatSession:
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


def _msgs(*pairs: tuple[str, str]) -> list[ChatMessage]:
    """Build messages with sequences 0..N-1 mirroring the production schema."""
    return [
        ChatMessage(role=r, content=c, sequence=i) for i, (r, c) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Scenarios P-R: the first SDK turn with no CLI session to resume, and the next
# ---------------------------------------------------------------------------


class TestFirstTurnWithoutACliSession:
    """The first SDK turn after N earlier turns that left no CLI session
    file.  ``_build_query_message`` must inject the complete prior session
    so the model has full context.
    """

    @pytest.mark.asyncio
    async def test_scenario_p_full_session_injected_on_the_first_turn(
        self, monkeypatch
    ):
        """Scenario P: the first SDK turn injects every earlier turn."""
        # 4 earlier messages (2 turns) followed by the first SDK turn.
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "earlier-q2"),
                ("assistant", "earlier-a2"),
                ("user", "sdk-q1"),  # current SDK turn
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        # transcript_msg_count=4: a transcript covers all 4 prior messages,
        # but use_resume=False (no CLI session to resume).
        result, stats = await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=4,
            session_id="s",
        )

        # All earlier turns must appear — none of them can be silently dropped.
        assert "<conversation_history>" in result
        assert "earlier-q1" in result
        assert "earlier-a1" in result
        assert "earlier-q2" in result
        assert "earlier-a2" in result
        assert "Now, the user says:\nsdk-q1" in result
        assert stats is None

    @pytest.mark.asyncio
    async def test_scenario_r_single_earlier_turn_injected(self, monkeypatch):
        """Scenario R: even a single earlier turn is captured on the first turn."""
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "sdk-q1"),
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        result, _ = await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=2,
            session_id="s",
        )

        assert "<conversation_history>" in result
        assert "earlier-q1" in result
        assert "earlier-a1" in result
        assert "Now, the user says:\nsdk-q1" in result

    @pytest.mark.asyncio
    async def test_scenario_q_the_next_turn_uses_resume(self):
        """Scenario Q: the turn after uses --resume.

        The first turn set session_id → the CLI wrote its session file →
        T2 restores the session → use_resume=True.  _build_query_message must
        return the bare message (--resume supplies context via native session).
        """
        # T2: 4 earlier messages + 1 SDK turn already recorded.
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "earlier-q2"),
                ("assistant", "earlier-a2"),
                ("user", "sdk-q1"),
                ("assistant", "sdk-a1"),
                ("user", "sdk-q2"),  # current SDK T2 message
            )
        )

        # transcript_msg_count=6 covers all prior messages → no gap.
        result, stats = await _build_query_message(
            "sdk-q2",
            session,
            use_resume=True,  # T2: --resume works after T1 set session_id
            transcript_msg_count=6,
            session_id="s",
        )

        # --resume has full context — bare message only.
        assert result == "sdk-q2"
        assert stats is None

    @pytest.mark.asyncio
    async def test_the_first_turn_compresses_every_earlier_turn(self, monkeypatch):
        """_compress_messages is called with ALL prior messages.

        There is exactly one compression call containing all 4 earlier messages
        — not just the 2 post-transcript-end messages.
        """
        session = _make_session(
            _msgs(
                ("user", "earlier-q1"),
                ("assistant", "earlier-a1"),
                ("user", "earlier-q2"),
                ("assistant", "earlier-a2"),
                ("user", "sdk-q1"),
            )
        )
        compressed_batches: list[list] = []

        async def _mock_compress(msgs, target_tokens=None):
            compressed_batches.append(list(msgs))
            return msgs, False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=4,
            session_id="s",
        )

        # Exactly one compression call, with all 4 prior messages.
        assert len(compressed_batches) == 1
        assert len(compressed_batches[0]) == 4
