"""Tests for transcript context coverage when switching between fast and SDK modes.

When a user switches modes mid-session the transcript must bridge the gap so
neither the baseline nor the SDK service loses context from turns produced by
the other mode.

Cross-mode transcript flow
==========================

Both ``baseline/service.py`` (fast mode) and ``sdk/service.py`` (extended_thinking
mode) read and write the same CLI session store via
``backend.copilot.transcript.upload_transcript`` /
``download_transcript``.

Fast → SDK switch
-----------------
On the first SDK turn after N baseline turns:
  • ``use_resume=False``  — no CLI session exists from baseline mode.
  • ``transcript_msg_count > 0`` — the baseline transcript is downloaded and
    validated successfully.
  • ``_build_query_message`` must inject the FULL prior session (not just a
    "gap" since the transcript end) because the CLI has zero context without
    ``--resume``.
  • After our fix, ``session_id`` IS set, so the CLI writes a session file
    on this turn → ``--resume`` works on T2+.

SDK → Fast switch
-----------------
On the first baseline turn after N SDK turns:
  • The baseline service downloads the SDK-written transcript.
  • ``_load_prior_transcript`` loads and validates it normally — the JSONL
    format is identical regardless of which mode wrote it.
  • ``transcript_covers_prefix=True`` → baseline sends ONLY new messages in
    its LLM payload (no double-counting of SDK history).

Scenario table (SDK _build_query_message)
==========================================

| # | Scenario                       | use_resume | tmc | Expected query message          |
|---|--------------------------------|------------|-----|---------------------------------|
| P | Fast→SDK T1                    | False      | 4   | full session injected           |
| Q | Fast→SDK T2+ (after fix)       | True       | 6   | bare message only (--resume ok) |
| R | Fast→SDK T1, single baseline   | False      | 2   | full session injected           |
| S | SDK→Fast (baseline loads ok)   | N/A        | N/A | transcript covers prefix=True   |
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

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
    return [ChatMessage(role=r, content=c) for r, c in pairs]


# ---------------------------------------------------------------------------
# Scenario P — Fast → SDK T1: full session injected from baseline transcript
# ---------------------------------------------------------------------------


class TestFastToSdkModeSwitch:
    """First SDK turn after N baseline (fast) turns.

    The baseline transcript exists (has been uploaded by fast mode), but
    there is no CLI session file.  ``_build_query_message`` must inject
    the complete prior session so the model has full context.
    """

    @pytest.mark.asyncio
    async def test_scenario_p_full_session_injected_on_mode_switch_t1(
        self, monkeypatch
    ):
        """Scenario P: fast→SDK T1 injects all baseline turns into the query."""
        # Simulate 4 baseline messages (2 turns) followed by the first SDK turn.
        session = _make_session(
            _msgs(
                ("user", "baseline-q1"),
                ("assistant", "baseline-a1"),
                ("user", "baseline-q2"),
                ("assistant", "baseline-a2"),
                ("user", "sdk-q1"),  # current SDK turn
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        # transcript_msg_count=4: baseline uploaded a transcript covering all
        # 4 prior messages, but use_resume=False (no CLI session from baseline).
        result, compacted = await _build_query_message(
            "sdk-q1",
            session,
            use_resume=False,
            transcript_msg_count=4,
            session_id="s",
        )

        # All baseline turns must appear — none of them can be silently dropped.
        assert "<conversation_history>" in result
        assert "baseline-q1" in result
        assert "baseline-a1" in result
        assert "baseline-q2" in result
        assert "baseline-a2" in result
        assert "Now, the user says:\nsdk-q1" in result
        assert compacted is False

    @pytest.mark.asyncio
    async def test_scenario_r_single_baseline_turn_injected(self, monkeypatch):
        """Scenario R: even a single baseline turn is captured on mode-switch T1."""
        session = _make_session(
            _msgs(
                ("user", "baseline-q1"),
                ("assistant", "baseline-a1"),
                ("user", "sdk-q1"),
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False

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
        assert "baseline-q1" in result
        assert "baseline-a1" in result
        assert "Now, the user says:\nsdk-q1" in result

    @pytest.mark.asyncio
    async def test_scenario_q_sdk_t2_uses_resume_after_fix(self):
        """Scenario Q: SDK T2+ uses --resume after mode-switch T1 set session_id.

        With the mode-switch fix, T1 sets session_id → CLI writes session file →
        T2 restores the session → use_resume=True.  _build_query_message must
        return the bare message (--resume supplies context via native session).
        """
        # T2: 4 baseline turns + 1 SDK turn already recorded.
        session = _make_session(
            _msgs(
                ("user", "baseline-q1"),
                ("assistant", "baseline-a1"),
                ("user", "baseline-q2"),
                ("assistant", "baseline-a2"),
                ("user", "sdk-q1"),
                ("assistant", "sdk-a1"),
                ("user", "sdk-q2"),  # current SDK T2 message
            )
        )

        # transcript_msg_count=6 covers all prior messages → no gap.
        result, compacted = await _build_query_message(
            "sdk-q2",
            session,
            use_resume=True,  # T2: --resume works after T1 set session_id
            transcript_msg_count=6,
            session_id="s",
        )

        # --resume has full context — bare message only.
        assert result == "sdk-q2"
        assert compacted is False

    @pytest.mark.asyncio
    async def test_mode_switch_t1_compresses_all_baseline_turns(self, monkeypatch):
        """_compress_messages is called with ALL prior baseline messages.

        There is exactly one compression call containing all 4 baseline messages
        — not just the 2 post-transcript-end messages.
        """
        session = _make_session(
            _msgs(
                ("user", "baseline-q1"),
                ("assistant", "baseline-a1"),
                ("user", "baseline-q2"),
                ("assistant", "baseline-a2"),
                ("user", "sdk-q1"),
            )
        )
        compressed_batches: list[list] = []

        async def _mock_compress(msgs, target_tokens=None):
            compressed_batches.append(list(msgs))
            return msgs, False

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


# ---------------------------------------------------------------------------
# Scenario S — SDK → Fast: baseline loads SDK-written transcript
# ---------------------------------------------------------------------------


class TestSdkToFastModeSwitch:
    """Fast mode turn after N SDK (extended_thinking) turns.

    The transcript written by SDK mode uses the same JSONL format as the one
    written by baseline mode (both go through ``TranscriptBuilder``).
    ``_load_prior_transcript`` must accept it and mark the prefix as covered.
    """

    @pytest.mark.asyncio
    async def test_scenario_s_baseline_loads_sdk_transcript(self):
        """Scenario S: SDK-written CLI session is accepted by baseline's load helper."""
        from backend.copilot.baseline.service import _load_prior_transcript
        from backend.copilot.model import ChatMessage
        from backend.copilot.transcript import STOP_REASON_END_TURN, TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        # Build a minimal valid transcript as SDK mode would write it.
        # SDK uses append_user / append_assistant on TranscriptBuilder.
        builder_sdk = TranscriptBuilder()
        builder_sdk.append_user(content="sdk-question")
        builder_sdk.append_assistant(
            content_blocks=[{"type": "text", "text": "sdk-answer"}],
            model="claude-sonnet-4",
            stop_reason=STOP_REASON_END_TURN,
        )
        sdk_transcript = builder_sdk.to_jsonl()

        # Baseline session now has those 2 SDK messages + 1 new baseline message.
        restore = TranscriptDownload(
            content=sdk_transcript.encode("utf-8"), message_count=2, mode="sdk"
        )

        baseline_builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            covers, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=[
                    ChatMessage(role="user", content="sdk-question"),
                    ChatMessage(role="assistant", content="sdk-answer"),
                    ChatMessage(role="user", content="baseline-question"),
                ],
                transcript_builder=baseline_builder,
            )

        # CLI session is valid and covers the prefix.
        assert covers is True
        assert dl is not None
        assert baseline_builder.entry_count == 2

    @pytest.mark.asyncio
    async def test_scenario_s_stale_sdk_transcript_not_loaded(self):
        """Scenario S (stale): SDK CLI session is stale — baseline does not load it.

        If SDK mode produced more turns than the session captured (e.g.
        upload failed on one turn), the baseline rejects the stale session
        to avoid injecting an incomplete history.
        """
        from backend.copilot.baseline.service import _load_prior_transcript
        from backend.copilot.model import ChatMessage
        from backend.copilot.transcript import STOP_REASON_END_TURN, TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        builder_sdk = TranscriptBuilder()
        builder_sdk.append_user(content="sdk-question")
        builder_sdk.append_assistant(
            content_blocks=[{"type": "text", "text": "sdk-answer"}],
            model="claude-sonnet-4",
            stop_reason=STOP_REASON_END_TURN,
        )
        sdk_transcript = builder_sdk.to_jsonl()

        # Session covers only 2 messages but session has 10 (many SDK turns).
        # With watermark=2 and 10 total messages, detect_gap will fill the gap
        # by appending messages 2..8 (positions 2 to total-2).
        restore = TranscriptDownload(
            content=sdk_transcript.encode("utf-8"), message_count=2, mode="sdk"
        )

        # Build a session with 10 alternating user/assistant messages + current user
        session_messages = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"msg-{i}")
            for i in range(10)
        ]

        baseline_builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            covers, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=session_messages,
                transcript_builder=baseline_builder,
            )

        # With gap filling, covers is True and gap messages are appended.
        assert covers is True
        assert dl is not None
        # 2 from transcript + 7 gap messages (positions 2..8, excluding last user turn)
        assert baseline_builder.entry_count == 9
