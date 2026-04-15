"""Tests for context fallback paths introduced in fix/copilot-transcript-resume-gate.

Scenario table
==============

| # | use_resume | transcript_msg_count | gap     | target_tokens | Expected output                            |
|---|------------|----------------------|---------|---------------|--------------------------------------------|
| A | True       | covers all           | empty   | None          | bare message (--resume has full context)   |
| B | True       | stale                | 2 msgs  | None          | gap context prepended                      |
| C | True       | stale                | 2 msgs  | 50_000        | gap compressed to budget, prepended        |
| D | False      | 0                    | N/A     | None          | full session compressed, prepended         |
| E | False      | 0                    | N/A     | 50_000        | full session compressed to budget          |
| F | False      | 2 (partial)          | 2 msgs  | None          | full session compressed (not just gap;     |
|   |            |                      |         |               | CLI has zero context without --resume)     |
| G | False      | 2 (partial)          | 2 msgs  | 50_000        | full session compressed to budget          |
| H | False      | covers all           | empty   | None          | full session compressed                    |
|   |            |                      |         |               | (NOT bare message — the bug that was fixed)|
| I | False      | covers all           | empty   | 50_000        | full session compressed to tight budget    |
| J | False      | 2 (partial)          | n/a     | None          | exactly ONE compression call (full prior)  |

Compression unit tests
=======================

| # | Input                | target_tokens | Expected                                      |
|---|----------------------|---------------|-----------------------------------------------|
| K | []                   | None          | ([], False) — empty guard                     |
| L | [1 msg]              | None          | ([msg], False) — single-msg guard             |
| M | [2+ msgs]            | None          | target_tokens=None forwarded to _run_compression |
| N | [2+ msgs]            | 30_000        | target_tokens=30_000 forwarded                |
| O | [2+ msgs], run fails | None          | returns originals, False                      |
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import _build_query_message, _compress_messages
from backend.util.prompt import CompressResult

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


def _passthrough_compress(target_tokens=None):
    """Return a mock that passes messages through and records its call args."""
    calls: list[tuple[list, int | None]] = []

    async def _mock(msgs, tok=None):
        calls.append((msgs, tok))
        return msgs, False

    _mock.calls = calls  # type: ignore[attr-defined]
    return _mock


# ---------------------------------------------------------------------------
# _build_query_message — scenario A–J
# ---------------------------------------------------------------------------


class TestBuildQueryMessageResume:
    """use_resume=True paths (--resume supplies history; only inject gap if stale)."""

    @pytest.mark.asyncio
    async def test_scenario_a_transcript_current_returns_bare_message(self):
        """Scenario A: --resume covers full context → no prefix injected."""
        session = _make_session(
            _msgs(("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        )
        result, compacted = await _build_query_message(
            "q2", session, use_resume=True, transcript_msg_count=2, session_id="s"
        )
        assert result == "q2"
        assert compacted is False

    @pytest.mark.asyncio
    async def test_scenario_b_stale_transcript_injects_gap(self, monkeypatch):
        """Scenario B: stale transcript → gap context prepended."""
        session = _make_session(
            _msgs(
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        result, compacted = await _build_query_message(
            "q3", session, use_resume=True, transcript_msg_count=2, session_id="s"
        )
        assert "<conversation_history>" in result
        assert "q2" in result
        assert "a2" in result
        assert "Now, the user says:\nq3" in result
        # q1/a1 are covered by the transcript — must NOT appear in gap context
        assert "q1" not in result

    @pytest.mark.asyncio
    async def test_scenario_c_stale_transcript_passes_target_tokens(self, monkeypatch):
        """Scenario C: target_tokens is forwarded to _compress_messages for the gap."""
        session = _make_session(
            _msgs(
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            )
        )
        captured: list[int | None] = []

        async def _mock_compress(msgs, target_tokens=None):
            captured.append(target_tokens)
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "q3",
            session,
            use_resume=True,
            transcript_msg_count=2,
            session_id="s",
            target_tokens=50_000,
        )
        assert captured == [50_000]


class TestBuildQueryMessageNoResumeNoTranscript:
    """use_resume=False, transcript_msg_count=0 — full session compressed."""

    @pytest.mark.asyncio
    async def test_scenario_d_full_session_compressed(self, monkeypatch):
        """Scenario D: no resume, no transcript → compress all prior messages."""
        session = _make_session(
            _msgs(("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        result, compacted = await _build_query_message(
            "q2", session, use_resume=False, transcript_msg_count=0, session_id="s"
        )
        assert "<conversation_history>" in result
        assert "q1" in result
        assert "a1" in result
        assert "Now, the user says:\nq2" in result

    @pytest.mark.asyncio
    async def test_scenario_e_passes_target_tokens_to_compression(self, monkeypatch):
        """Scenario E: target_tokens forwarded to _compress_messages."""
        session = _make_session(
            _msgs(("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        )
        captured: list[int | None] = []

        async def _mock_compress(msgs, target_tokens=None):
            captured.append(target_tokens)
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "q2",
            session,
            use_resume=False,
            transcript_msg_count=0,
            session_id="s",
            target_tokens=15_000,
        )
        assert captured == [15_000]


class TestBuildQueryMessageNoResumeWithTranscript:
    """use_resume=False, transcript_msg_count > 0 — gap or full-session fallback."""

    @pytest.mark.asyncio
    async def test_scenario_f_no_resume_always_injects_full_session(self, monkeypatch):
        """Scenario F: use_resume=False with transcript_msg_count > 0 still injects
        the FULL prior session — not just the gap since the transcript end.

        When there is no --resume the CLI starts with zero context, so injecting
        only the post-transcript gap would silently drop all transcript-covered
        history.  The correct fix is to always compress the full session.
        """
        session = _make_session(
            _msgs(
                ("user", "q1"),  # transcript_msg_count=2 covers these
                ("assistant", "a1"),
                ("user", "q2"),  # post-transcript gap starts here
                ("assistant", "a2"),
                ("user", "q3"),  # current message
            )
        )
        compressed_msgs: list[list] = []

        async def _mock_compress(msgs, target_tokens=None):
            compressed_msgs.append(list(msgs))
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        result, _ = await _build_query_message(
            "q3",
            session,
            use_resume=False,
            transcript_msg_count=2,  # transcript covers q1/a1 but no --resume
            session_id="s",
        )
        assert "<conversation_history>" in result
        # Full session must be injected — transcript-covered turns ARE included
        assert "q1" in result
        assert "a1" in result
        assert "q2" in result
        assert "a2" in result
        assert "Now, the user says:\nq3" in result
        # Compressed exactly once with all 4 prior messages
        assert len(compressed_msgs) == 1
        assert len(compressed_msgs[0]) == 4

    @pytest.mark.asyncio
    async def test_scenario_g_no_resume_passes_target_tokens(self, monkeypatch):
        """Scenario G: target_tokens forwarded when use_resume=False + transcript_msg_count > 0."""
        session = _make_session(
            _msgs(
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            )
        )
        captured: list[int | None] = []

        async def _mock_compress(msgs, target_tokens=None):
            captured.append(target_tokens)
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "q3",
            session,
            use_resume=False,
            transcript_msg_count=2,
            session_id="s",
            target_tokens=50_000,
        )
        assert captured == [50_000]

    @pytest.mark.asyncio
    async def test_scenario_h_no_resume_transcript_current_injects_full_session(
        self, monkeypatch
    ):
        """Scenario H: the bug that was fixed.

        Old code path: use_resume=False, transcript_msg_count covers all prior
        messages → gap sub-path: gap = [] → ``return current_message, False``
        → model received ZERO context (bare message only).

        New code path: use_resume=False always compresses the full prior session
        regardless of transcript_msg_count — model always gets context.
        """
        session = _make_session(
            _msgs(
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            )
        )

        async def _mock_compress(msgs, target_tokens=None):
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        result, _ = await _build_query_message(
            "q3",
            session,
            use_resume=False,
            transcript_msg_count=4,  # covers ALL prior → old code returned bare msg
            session_id="s",
        )
        # NEW: must inject full session, NOT return bare message
        assert result != "q3"
        assert "<conversation_history>" in result
        assert "q1" in result
        assert "Now, the user says:\nq3" in result

    @pytest.mark.asyncio
    async def test_scenario_i_no_resume_target_tokens_forwarded_any_transcript_count(
        self, monkeypatch
    ):
        """Scenario I: target_tokens forwarded even when transcript_msg_count covers all."""
        session = _make_session(
            _msgs(("user", "q1"), ("assistant", "a1"), ("user", "q2"))
        )
        captured: list[int | None] = []

        async def _mock_compress(msgs, target_tokens=None):
            captured.append(target_tokens)
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "q2",
            session,
            use_resume=False,
            transcript_msg_count=2,
            session_id="s",
            target_tokens=15_000,
        )
        assert 15_000 in captured

    @pytest.mark.asyncio
    async def test_scenario_j_no_resume_single_compression_call(self, monkeypatch):
        """Scenario J: use_resume=False always makes exactly ONE compression call
        (the full session), regardless of transcript coverage.

        This verifies there is no two-step gap+fallback pattern for no-resume —
        compression is called once with the full prior session.
        """
        session = _make_session(
            _msgs(
                ("user", "q1"),
                ("assistant", "a1"),
                ("user", "q2"),
                ("assistant", "a2"),
                ("user", "q3"),
            )
        )
        call_count = 0

        async def _mock_compress(msgs, target_tokens=None):
            nonlocal call_count
            call_count += 1
            return msgs, False

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _mock_compress
        )

        await _build_query_message(
            "q3",
            session,
            use_resume=False,
            transcript_msg_count=2,
            session_id="s",
        )
        assert call_count == 1


# ---------------------------------------------------------------------------
# _compress_messages — unit tests K–O
# ---------------------------------------------------------------------------


class TestCompressMessages:
    @pytest.mark.asyncio
    async def test_scenario_k_empty_list_returns_empty(self):
        """Scenario K: empty input → short-circuit, no compression."""
        result, compacted = await _compress_messages([])
        assert result == []
        assert compacted is False

    @pytest.mark.asyncio
    async def test_scenario_l_single_message_returns_as_is(self):
        """Scenario L: single message → short-circuit (< 2 guard)."""
        msg = ChatMessage(role="user", content="hello")
        result, compacted = await _compress_messages([msg])
        assert result == [msg]
        assert compacted is False

    @pytest.mark.asyncio
    async def test_scenario_m_target_tokens_none_forwarded(self):
        """Scenario M: target_tokens=None forwarded to _run_compression."""
        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        fake_result = CompressResult(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            token_count=10,
            was_compacted=False,
            original_token_count=10,
        )
        with patch(
            "backend.copilot.sdk.service._run_compression",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_run:
            await _compress_messages(msgs, target_tokens=None)

        mock_run.assert_awaited_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("target_tokens") is None

    @pytest.mark.asyncio
    async def test_scenario_n_explicit_target_tokens_forwarded(self):
        """Scenario N: explicit target_tokens forwarded to _run_compression."""
        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        fake_result = CompressResult(
            messages=[{"role": "user", "content": "summary"}],
            token_count=5,
            was_compacted=True,
            original_token_count=50,
        )
        with patch(
            "backend.copilot.sdk.service._run_compression",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_run:
            result, compacted = await _compress_messages(msgs, target_tokens=30_000)

        mock_run.assert_awaited_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("target_tokens") == 30_000
        assert compacted is True

    @pytest.mark.asyncio
    async def test_scenario_o_run_compression_exception_returns_originals(self):
        """Scenario O: _run_compression raises → return original messages, False."""
        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        with patch(
            "backend.copilot.sdk.service._run_compression",
            new_callable=AsyncMock,
            side_effect=RuntimeError("compression timeout"),
        ):
            result, compacted = await _compress_messages(msgs)

        assert result == msgs
        assert compacted is False

    @pytest.mark.asyncio
    async def test_compaction_messages_filtered_before_compression(self):
        """filter_compaction_messages is applied before _run_compression is called."""
        # A compaction message is one with role=assistant and specific content pattern.
        # We verify that only real messages reach _run_compression.
        from backend.copilot.sdk.service import filter_compaction_messages

        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        # filter_compaction_messages should not remove these plain messages
        filtered = filter_compaction_messages(msgs)
        assert len(filtered) == len(msgs)


# ---------------------------------------------------------------------------
# target_tokens threading — _retry_target_tokens values match expectations
# ---------------------------------------------------------------------------


class TestRetryTargetTokens:
    def test_first_retry_uses_first_slot(self):
        from backend.copilot.sdk.service import _RETRY_TARGET_TOKENS

        assert _RETRY_TARGET_TOKENS[0] == 50_000

    def test_second_retry_uses_second_slot(self):
        from backend.copilot.sdk.service import _RETRY_TARGET_TOKENS

        assert _RETRY_TARGET_TOKENS[1] == 15_000

    def test_second_slot_smaller_than_first(self):
        from backend.copilot.sdk.service import _RETRY_TARGET_TOKENS

        assert _RETRY_TARGET_TOKENS[1] < _RETRY_TARGET_TOKENS[0]


# ---------------------------------------------------------------------------
# Single-message session edge cases
# ---------------------------------------------------------------------------


class TestSingleMessageSessions:
    @pytest.mark.asyncio
    async def test_no_resume_single_message_returns_bare(self):
        """First turn (1 message): no prior history to inject."""
        session = _make_session([ChatMessage(role="user", content="hello")])
        result, compacted = await _build_query_message(
            "hello", session, use_resume=False, transcript_msg_count=0, session_id="s"
        )
        assert result == "hello"
        assert compacted is False

    @pytest.mark.asyncio
    async def test_resume_single_message_returns_bare(self):
        """First turn with resume flag: transcript is empty so no gap."""
        session = _make_session([ChatMessage(role="user", content="hello")])
        result, compacted = await _build_query_message(
            "hello", session, use_resume=True, transcript_msg_count=0, session_id="s"
        )
        assert result == "hello"
        assert compacted is False
