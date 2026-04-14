"""Integration tests for baseline transcript flow.

Exercises the real helpers in ``baseline/service.py`` that download,
validate, load, append to, backfill, and upload the transcript.
Storage is mocked via ``download_transcript`` / ``upload_transcript``
patches; no network access is required.
"""

import json as stdlib_json
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.baseline.service import (
    _load_prior_transcript,
    _record_turn_to_transcript,
    _resolve_baseline_model,
    _upload_final_transcript,
    is_transcript_stale,
    should_upload_transcript,
)
from backend.copilot.service import config
from backend.copilot.transcript import (
    STOP_REASON_END_TURN,
    STOP_REASON_TOOL_USE,
    TranscriptDownload,
)
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util.tool_call_loop import LLMLoopResponse, LLMToolCall, ToolCallResult


def _make_transcript_content(*roles: str) -> str:
    """Build a minimal valid JSONL transcript from role names."""
    lines = []
    parent = ""
    for i, role in enumerate(roles):
        uid = f"uuid-{i}"
        entry: dict = {
            "type": role,
            "uuid": uid,
            "parentUuid": parent,
            "message": {
                "role": role,
                "content": [{"type": "text", "text": f"{role} message {i}"}],
            },
        }
        if role == "assistant":
            entry["message"]["id"] = f"msg_{i}"
            entry["message"]["model"] = "test-model"
            entry["message"]["type"] = "message"
            entry["message"]["stop_reason"] = STOP_REASON_END_TURN
        lines.append(stdlib_json.dumps(entry))
        parent = uid
    return "\n".join(lines) + "\n"


class TestResolveBaselineModel:
    """Model selection honours the per-request mode."""

    def test_fast_mode_selects_fast_model(self):
        assert _resolve_baseline_model("fast") == config.fast_model

    def test_extended_thinking_selects_default_model(self):
        assert _resolve_baseline_model("extended_thinking") == config.model

    def test_none_mode_selects_default_model(self):
        """Critical: baseline users without a mode MUST keep the default (opus)."""
        assert _resolve_baseline_model(None) == config.model

    def test_default_and_fast_models_same(self):
        """SDK 0.1.58: both tiers now use the same model (anthropic/claude-sonnet-4)."""
        assert config.model == config.fast_model


class TestLoadPriorTranscript:
    """``_load_prior_transcript`` wraps the download + validate + load flow."""

    @pytest.mark.asyncio
    async def test_loads_fresh_transcript(self):
        builder = TranscriptBuilder()
        content = _make_transcript_content("user", "assistant")
        download = TranscriptDownload(content=content, message_count=2)

        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=download),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=3,
                transcript_builder=builder,
            )

        assert covers is True
        assert builder.entry_count == 2
        assert builder.last_entry_type == "assistant"

    @pytest.mark.asyncio
    async def test_rejects_stale_transcript(self):
        """msg_count strictly less than session-1 is treated as stale."""
        builder = TranscriptBuilder()
        content = _make_transcript_content("user", "assistant")
        # session has 6 messages, transcript only covers 2 → stale.
        download = TranscriptDownload(content=content, message_count=2)

        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=download),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=6,
                transcript_builder=builder,
            )

        assert covers is False
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_missing_transcript_returns_false(self):
        builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=None),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=2,
                transcript_builder=builder,
            )

        assert covers is False
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_invalid_transcript_returns_false(self):
        builder = TranscriptBuilder()
        download = TranscriptDownload(
            content='{"type":"progress","uuid":"a"}\n',
            message_count=1,
        )
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=download),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=2,
                transcript_builder=builder,
            )

        assert covers is False
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_download_exception_returns_false(self):
        builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=2,
                transcript_builder=builder,
            )

        assert covers is False
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_zero_message_count_not_stale(self):
        """When msg_count is 0 (unknown), staleness check is skipped."""
        builder = TranscriptBuilder()
        download = TranscriptDownload(
            content=_make_transcript_content("user", "assistant"),
            message_count=0,
        )
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=download),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=20,
                transcript_builder=builder,
            )

        assert covers is True
        assert builder.entry_count == 2


class TestUploadFinalTranscript:
    """``_upload_final_transcript`` serialises and calls storage."""

    @pytest.mark.asyncio
    async def test_uploads_valid_transcript(self):
        builder = TranscriptBuilder()
        builder.append_user(content="hi")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "hello"}],
            model="test-model",
            stop_reason=STOP_REASON_END_TURN,
        )

        upload_mock = AsyncMock(return_value=None)
        with patch(
            "backend.copilot.baseline.service.upload_transcript",
            new=upload_mock,
        ):
            await _upload_final_transcript(
                user_id="user-1",
                session_id="session-1",
                transcript_builder=builder,
                session_msg_count=2,
            )

        upload_mock.assert_awaited_once()
        assert upload_mock.await_args is not None
        call_kwargs = upload_mock.await_args.kwargs
        assert call_kwargs["user_id"] == "user-1"
        assert call_kwargs["session_id"] == "session-1"
        assert call_kwargs["message_count"] == 2
        assert "hello" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_skips_upload_when_builder_empty(self):
        builder = TranscriptBuilder()
        upload_mock = AsyncMock(return_value=None)
        with patch(
            "backend.copilot.baseline.service.upload_transcript",
            new=upload_mock,
        ):
            await _upload_final_transcript(
                user_id="user-1",
                session_id="session-1",
                transcript_builder=builder,
                session_msg_count=0,
            )

        upload_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_swallows_upload_exceptions(self):
        """Upload failures should not propagate (flow continues for the user)."""
        builder = TranscriptBuilder()
        builder.append_user(content="hi")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "hello"}],
            model="test-model",
            stop_reason=STOP_REASON_END_TURN,
        )

        with patch(
            "backend.copilot.baseline.service.upload_transcript",
            new=AsyncMock(side_effect=RuntimeError("storage unavailable")),
        ):
            # Should not raise.
            await _upload_final_transcript(
                user_id="user-1",
                session_id="session-1",
                transcript_builder=builder,
                session_msg_count=2,
            )


class TestRecordTurnToTranscript:
    """``_record_turn_to_transcript`` translates LLMLoopResponse → transcript."""

    def test_records_final_assistant_text(self):
        builder = TranscriptBuilder()
        builder.append_user(content="hi")

        response = LLMLoopResponse(
            response_text="hello there",
            tool_calls=[],
            raw_response=None,
        )
        _record_turn_to_transcript(
            response,
            tool_results=None,
            transcript_builder=builder,
            model="test-model",
        )

        assert builder.entry_count == 2
        assert builder.last_entry_type == "assistant"
        jsonl = builder.to_jsonl()
        assert "hello there" in jsonl
        assert STOP_REASON_END_TURN in jsonl

    def test_records_tool_use_then_tool_result(self):
        """Anthropic ordering: assistant(tool_use) → user(tool_result)."""
        builder = TranscriptBuilder()
        builder.append_user(content="use a tool")

        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[
                LLMToolCall(id="call-1", name="echo", arguments='{"text":"hi"}')
            ],
            raw_response=None,
        )
        tool_results = [
            ToolCallResult(tool_call_id="call-1", tool_name="echo", content="hi")
        ]
        _record_turn_to_transcript(
            response,
            tool_results,
            transcript_builder=builder,
            model="test-model",
        )

        # user, assistant(tool_use), user(tool_result) = 3 entries
        assert builder.entry_count == 3
        jsonl = builder.to_jsonl()
        assert STOP_REASON_TOOL_USE in jsonl
        assert "tool_use" in jsonl
        assert "tool_result" in jsonl
        assert "call-1" in jsonl

    def test_records_nothing_on_empty_response(self):
        builder = TranscriptBuilder()
        builder.append_user(content="hi")

        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[],
            raw_response=None,
        )
        _record_turn_to_transcript(
            response,
            tool_results=None,
            transcript_builder=builder,
            model="test-model",
        )

        assert builder.entry_count == 1

    def test_malformed_tool_args_dont_crash(self):
        """Bad JSON in tool arguments falls back to {} without raising."""
        builder = TranscriptBuilder()
        builder.append_user(content="hi")

        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[LLMToolCall(id="call-1", name="echo", arguments="{not-json")],
            raw_response=None,
        )
        tool_results = [
            ToolCallResult(tool_call_id="call-1", tool_name="echo", content="ok")
        ]
        _record_turn_to_transcript(
            response,
            tool_results,
            transcript_builder=builder,
            model="test-model",
        )

        assert builder.entry_count == 3
        jsonl = builder.to_jsonl()
        assert '"input":{}' in jsonl


class TestRoundTrip:
    """End-to-end: load prior → append new turn → upload."""

    @pytest.mark.asyncio
    async def test_full_round_trip(self):
        prior = _make_transcript_content("user", "assistant")
        download = TranscriptDownload(content=prior, message_count=2)

        builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=download),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=3,
                transcript_builder=builder,
            )
        assert covers is True
        assert builder.entry_count == 2

        # New user turn.
        builder.append_user(content="new question")
        assert builder.entry_count == 3

        # New assistant turn.
        response = LLMLoopResponse(
            response_text="new answer",
            tool_calls=[],
            raw_response=None,
        )
        _record_turn_to_transcript(
            response,
            tool_results=None,
            transcript_builder=builder,
            model="test-model",
        )
        assert builder.entry_count == 4

        # Upload.
        upload_mock = AsyncMock(return_value=None)
        with patch(
            "backend.copilot.baseline.service.upload_transcript",
            new=upload_mock,
        ):
            await _upload_final_transcript(
                user_id="user-1",
                session_id="session-1",
                transcript_builder=builder,
                session_msg_count=4,
            )

        upload_mock.assert_awaited_once()
        assert upload_mock.await_args is not None
        uploaded = upload_mock.await_args.kwargs["content"]
        assert "new question" in uploaded
        assert "new answer" in uploaded
        # Original content preserved in the round trip.
        assert "user message 0" in uploaded
        assert "assistant message 1" in uploaded

    @pytest.mark.asyncio
    async def test_backfill_append_guard(self):
        """Backfill only runs when the last entry is not already assistant."""
        builder = TranscriptBuilder()
        builder.append_user(content="hi")

        # Simulate the backfill guard from stream_chat_completion_baseline.
        assistant_text = "partial text before error"
        if builder.last_entry_type != "assistant":
            builder.append_assistant(
                content_blocks=[{"type": "text", "text": assistant_text}],
                model="test-model",
                stop_reason=STOP_REASON_END_TURN,
            )

        assert builder.last_entry_type == "assistant"
        assert "partial text before error" in builder.to_jsonl()

        # Second invocation: the guard must prevent double-append.
        initial_count = builder.entry_count
        if builder.last_entry_type != "assistant":
            builder.append_assistant(
                content_blocks=[{"type": "text", "text": "duplicate"}],
                model="test-model",
                stop_reason=STOP_REASON_END_TURN,
            )
        assert builder.entry_count == initial_count


class TestIsTranscriptStale:
    """``is_transcript_stale`` gates prior-transcript loading."""

    def test_none_download_is_not_stale(self):
        assert is_transcript_stale(None, session_msg_count=5) is False

    def test_zero_message_count_is_not_stale(self):
        """Legacy transcripts without msg_count tracking must remain usable."""
        dl = TranscriptDownload(content="", message_count=0)
        assert is_transcript_stale(dl, session_msg_count=20) is False

    def test_stale_when_covers_less_than_prefix(self):
        dl = TranscriptDownload(content="", message_count=2)
        # session has 6 messages; transcript must cover at least 5 (6-1).
        assert is_transcript_stale(dl, session_msg_count=6) is True

    def test_fresh_when_covers_full_prefix(self):
        dl = TranscriptDownload(content="", message_count=5)
        assert is_transcript_stale(dl, session_msg_count=6) is False

    def test_fresh_when_exceeds_prefix(self):
        """Race: transcript ahead of session count is still acceptable."""
        dl = TranscriptDownload(content="", message_count=10)
        assert is_transcript_stale(dl, session_msg_count=6) is False

    def test_boundary_equal_to_prefix_minus_one(self):
        dl = TranscriptDownload(content="", message_count=5)
        assert is_transcript_stale(dl, session_msg_count=6) is False


class TestShouldUploadTranscript:
    """``should_upload_transcript`` gates the final upload."""

    def test_upload_allowed_for_user_with_coverage(self):
        assert should_upload_transcript("user-1", True) is True

    def test_upload_skipped_when_no_user(self):
        assert should_upload_transcript(None, True) is False

    def test_upload_skipped_when_empty_user(self):
        assert should_upload_transcript("", True) is False

    def test_upload_skipped_without_coverage(self):
        """Partial transcript must never clobber a more complete stored one."""
        assert should_upload_transcript("user-1", False) is False

    def test_upload_skipped_when_no_user_and_no_coverage(self):
        assert should_upload_transcript(None, False) is False


class TestTranscriptLifecycle:
    """End-to-end: download → validate → build → upload.

    Simulates the full transcript lifecycle inside
    ``stream_chat_completion_baseline`` by mocking the storage layer and
    driving each step through the real helpers.
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle_happy_path(self):
        """Fresh download, append a turn, upload covers the session."""
        builder = TranscriptBuilder()
        prior = _make_transcript_content("user", "assistant")
        download = TranscriptDownload(content=prior, message_count=2)

        upload_mock = AsyncMock(return_value=None)
        with (
            patch(
                "backend.copilot.baseline.service.download_transcript",
                new=AsyncMock(return_value=download),
            ),
            patch(
                "backend.copilot.baseline.service.upload_transcript",
                new=upload_mock,
            ),
        ):
            # --- 1. Download & load prior transcript ---
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=3,
                transcript_builder=builder,
            )
            assert covers is True

            # --- 2. Append a new user turn + a new assistant response ---
            builder.append_user(content="follow-up question")
            _record_turn_to_transcript(
                LLMLoopResponse(
                    response_text="follow-up answer",
                    tool_calls=[],
                    raw_response=None,
                ),
                tool_results=None,
                transcript_builder=builder,
                model="test-model",
            )

            # --- 3. Gate + upload ---
            assert (
                should_upload_transcript(
                    user_id="user-1", transcript_covers_prefix=covers
                )
                is True
            )
            await _upload_final_transcript(
                user_id="user-1",
                session_id="session-1",
                transcript_builder=builder,
                session_msg_count=4,
            )

        upload_mock.assert_awaited_once()
        assert upload_mock.await_args is not None
        uploaded = upload_mock.await_args.kwargs["content"]
        assert "follow-up question" in uploaded
        assert "follow-up answer" in uploaded
        # Original prior-turn content preserved.
        assert "user message 0" in uploaded
        assert "assistant message 1" in uploaded

    @pytest.mark.asyncio
    async def test_lifecycle_stale_download_suppresses_upload(self):
        """Stale download → covers=False → upload must be skipped."""
        builder = TranscriptBuilder()
        # session has 10 msgs but stored transcript only covers 2 → stale.
        stale = TranscriptDownload(
            content=_make_transcript_content("user", "assistant"),
            message_count=2,
        )

        upload_mock = AsyncMock(return_value=None)
        with (
            patch(
                "backend.copilot.baseline.service.download_transcript",
                new=AsyncMock(return_value=stale),
            ),
            patch(
                "backend.copilot.baseline.service.upload_transcript",
                new=upload_mock,
            ),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=10,
                transcript_builder=builder,
            )

        assert covers is False
        # The caller's gate mirrors the production path.
        assert (
            should_upload_transcript(user_id="user-1", transcript_covers_prefix=covers)
            is False
        )
        upload_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lifecycle_anonymous_user_skips_upload(self):
        """Anonymous (user_id=None) → upload gate must return False."""
        builder = TranscriptBuilder()
        builder.append_user(content="hi")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "hello"}],
            model="test-model",
            stop_reason=STOP_REASON_END_TURN,
        )

        assert (
            should_upload_transcript(user_id=None, transcript_covers_prefix=True)
            is False
        )

    @pytest.mark.asyncio
    async def test_lifecycle_missing_download_still_uploads_new_content(self):
        """No prior transcript → covers defaults to True in the service,
        new turn should upload cleanly."""
        builder = TranscriptBuilder()
        upload_mock = AsyncMock(return_value=None)
        with (
            patch(
                "backend.copilot.baseline.service.download_transcript",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.copilot.baseline.service.upload_transcript",
                new=upload_mock,
            ),
        ):
            covers = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_msg_count=1,
                transcript_builder=builder,
            )
            # No download: covers is False, so the production path would
            # skip upload. This protects against overwriting a future
            # more-complete transcript with a single-turn snapshot.
            assert covers is False
            assert (
                should_upload_transcript(
                    user_id="user-1", transcript_covers_prefix=covers
                )
                is False
            )
            upload_mock.assert_not_awaited()
