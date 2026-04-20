"""Integration tests for baseline transcript flow.

Exercises the real helpers in ``baseline/service.py`` that restore,
validate, load, append to, backfill, and upload the CLI session.
Storage is mocked via ``download_transcript`` / ``upload_transcript``
patches; no network access is required.
"""

import json as stdlib_json
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.baseline.service import (
    _append_gap_to_builder,
    _load_prior_transcript,
    _record_turn_to_transcript,
    _resolve_baseline_model,
    _upload_final_transcript,
    should_upload_transcript,
)
from backend.copilot.model import ChatMessage
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


def _make_session_messages(*roles: str) -> list[ChatMessage]:
    """Build a list of ChatMessage objects matching the given roles."""
    return [
        ChatMessage(role=r, content=f"{r} message {i}") for i, r in enumerate(roles)
    ]


class TestResolveBaselineModel:
    """Baseline model resolution honours the per-request tier toggle."""

    def test_advanced_tier_selects_advanced_model(self):
        assert _resolve_baseline_model("advanced") == config.advanced_model

    def test_standard_tier_selects_default_model(self):
        assert _resolve_baseline_model("standard") == config.model

    def test_none_tier_selects_default_model(self):
        """Baseline users without a tier MUST keep the default (standard)."""
        assert _resolve_baseline_model(None) == config.model

    def test_standard_and_advanced_models_differ(self):
        """Advanced tier defaults to a different (Opus) model than standard."""
        assert config.model != config.advanced_model


class TestLoadPriorTranscript:
    """``_load_prior_transcript`` wraps the CLI session restore + validate + load flow."""

    @pytest.mark.asyncio
    async def test_loads_fresh_transcript(self):
        builder = TranscriptBuilder()
        content = _make_transcript_content("user", "assistant")
        restore = TranscriptDownload(
            content=content.encode("utf-8"), message_count=2, mode="sdk"
        )

        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            covers, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user", "assistant", "user"),
                transcript_builder=builder,
            )

        assert covers is True
        assert dl is not None
        assert dl.message_count == 2
        assert builder.entry_count == 2
        assert builder.last_entry_type == "assistant"

    @pytest.mark.asyncio
    async def test_fills_gap_when_transcript_is_behind(self):
        """When transcript covers fewer messages than session, gap is filled from DB."""
        builder = TranscriptBuilder()
        content = _make_transcript_content("user", "assistant")
        # transcript covers 2 messages, session has 4 (plus current user turn = 5)
        restore = TranscriptDownload(
            content=content.encode("utf-8"), message_count=2, mode="baseline"
        )

        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            covers, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages(
                    "user", "assistant", "user", "assistant", "user"
                ),
                transcript_builder=builder,
            )

        assert covers is True
        assert dl is not None
        # 2 from transcript + 2 gap messages (user+assistant at positions 2,3)
        assert builder.entry_count == 4

    @pytest.mark.asyncio
    async def test_missing_transcript_allows_upload(self):
        """Nothing in GCS → upload is safe; the turn writes the first snapshot."""
        builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=None),
        ):
            upload_safe, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user", "assistant"),
                transcript_builder=builder,
            )

        assert upload_safe is True
        assert dl is None
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_invalid_transcript_allows_upload(self):
        """Corrupt file in GCS → overwriting with a valid one is better."""
        builder = TranscriptBuilder()
        restore = TranscriptDownload(
            content=b'{"type":"progress","uuid":"a"}\n',
            message_count=1,
            mode="sdk",
        )
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            upload_safe, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user", "assistant"),
                transcript_builder=builder,
            )

        assert upload_safe is True
        assert dl is None
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_download_exception_returns_false(self):
        builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            covers, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user", "assistant"),
                transcript_builder=builder,
            )

        assert covers is False
        assert dl is None
        assert builder.is_empty

    @pytest.mark.asyncio
    async def test_zero_message_count_not_stale(self):
        """When msg_count is 0 (unknown), gap detection is skipped."""
        builder = TranscriptBuilder()
        restore = TranscriptDownload(
            content=_make_transcript_content("user", "assistant").encode("utf-8"),
            message_count=0,
            mode="sdk",
        )
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            covers, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages(*["user"] * 20),
                transcript_builder=builder,
            )

        assert covers is True
        assert dl is not None
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
        assert b"hello" in call_kwargs["content"]

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
        restore = TranscriptDownload(
            content=prior.encode("utf-8"), message_count=2, mode="sdk"
        )

        builder = TranscriptBuilder()
        with patch(
            "backend.copilot.baseline.service.download_transcript",
            new=AsyncMock(return_value=restore),
        ):
            covers, _ = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user", "assistant", "user"),
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
        assert b"new question" in uploaded
        assert b"new answer" in uploaded
        # Original content preserved in the round trip.
        assert b"user message 0" in uploaded
        assert b"assistant message 1" in uploaded

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
    """End-to-end: restore → validate → build → upload.

    Simulates the full transcript lifecycle inside
    ``stream_chat_completion_baseline`` by mocking the storage layer and
    driving each step through the real helpers.
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle_happy_path(self):
        """Fresh restore, append a turn, upload covers the session."""
        builder = TranscriptBuilder()
        prior = _make_transcript_content("user", "assistant")
        restore = TranscriptDownload(
            content=prior.encode("utf-8"), message_count=2, mode="sdk"
        )

        upload_mock = AsyncMock(return_value=None)
        with (
            patch(
                "backend.copilot.baseline.service.download_transcript",
                new=AsyncMock(return_value=restore),
            ),
            patch(
                "backend.copilot.baseline.service.upload_transcript",
                new=upload_mock,
            ),
        ):
            # --- 1. Restore & load prior session ---
            covers, _ = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user", "assistant", "user"),
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
                should_upload_transcript(user_id="user-1", upload_safe=covers) is True
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
        assert b"follow-up question" in uploaded
        assert b"follow-up answer" in uploaded
        # Original prior-turn content preserved.
        assert b"user message 0" in uploaded
        assert b"assistant message 1" in uploaded

    @pytest.mark.asyncio
    async def test_lifecycle_stale_download_fills_gap(self):
        """When transcript covers fewer messages, gap is filled rather than rejected."""
        builder = TranscriptBuilder()
        # session has 5 msgs but stored transcript only covers 2 → gap filled.
        stale = TranscriptDownload(
            content=_make_transcript_content("user", "assistant").encode("utf-8"),
            message_count=2,
            mode="baseline",
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
            covers, _ = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages(
                    "user", "assistant", "user", "assistant", "user"
                ),
                transcript_builder=builder,
            )

        assert covers is True
        # Gap was filled: 2 from transcript + 2 gap messages
        assert builder.entry_count == 4

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

        assert should_upload_transcript(user_id=None, upload_safe=True) is False

    @pytest.mark.asyncio
    async def test_lifecycle_missing_download_still_uploads_new_content(self):
        """No prior session → upload is safe; the turn writes the first snapshot."""
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
            upload_safe, dl = await _load_prior_transcript(
                user_id="user-1",
                session_id="session-1",
                session_messages=_make_session_messages("user"),
                transcript_builder=builder,
            )
            # Nothing in GCS → upload is safe so the first baseline turn
            # can write the initial transcript snapshot.
            assert upload_safe is True
            assert dl is None
            assert (
                should_upload_transcript(user_id="user-1", upload_safe=upload_safe)
                is True
            )


# ---------------------------------------------------------------------------
# _append_gap_to_builder
# ---------------------------------------------------------------------------


class TestAppendGapToBuilder:
    """``_append_gap_to_builder`` converts ChatMessage objects to TranscriptBuilder entries."""

    def test_user_message_appended(self):
        builder = TranscriptBuilder()
        msgs = [ChatMessage(role="user", content="hello")]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 1
        assert builder.last_entry_type == "user"

    def test_assistant_text_message_appended(self):
        builder = TranscriptBuilder()
        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="answer"),
        ]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 2
        assert builder.last_entry_type == "assistant"
        assert "answer" in builder.to_jsonl()

    def test_assistant_with_tool_calls_appended(self):
        """Assistant tool_calls are recorded as tool_use blocks in the transcript."""
        builder = TranscriptBuilder()
        tool_call = {
            "id": "tc-1",
            "type": "function",
            "function": {"name": "my_tool", "arguments": '{"key":"val"}'},
        }
        msgs = [ChatMessage(role="assistant", content=None, tool_calls=[tool_call])]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 1
        jsonl = builder.to_jsonl()
        assert "tool_use" in jsonl
        assert "my_tool" in jsonl
        assert "tc-1" in jsonl

    def test_assistant_invalid_json_args_uses_empty_dict(self):
        """Malformed JSON in tool_call arguments falls back to {}."""
        builder = TranscriptBuilder()
        tool_call = {
            "id": "tc-bad",
            "type": "function",
            "function": {"name": "bad_tool", "arguments": "not-json"},
        }
        msgs = [ChatMessage(role="assistant", content=None, tool_calls=[tool_call])]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 1
        jsonl = builder.to_jsonl()
        assert '"input":{}' in jsonl

    def test_assistant_empty_content_and_no_tools_uses_fallback(self):
        """Assistant with no content and no tool_calls gets a fallback empty text block."""
        builder = TranscriptBuilder()
        msgs = [ChatMessage(role="assistant", content=None)]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 1
        jsonl = builder.to_jsonl()
        assert "text" in jsonl

    def test_tool_role_with_tool_call_id_appended(self):
        """Tool result messages are appended when tool_call_id is set."""
        builder = TranscriptBuilder()
        # Need a preceding assistant tool_use entry
        builder.append_user("use tool")
        builder.append_assistant(
            content_blocks=[
                {"type": "tool_use", "id": "tc-1", "name": "my_tool", "input": {}}
            ]
        )
        msgs = [ChatMessage(role="tool", tool_call_id="tc-1", content="result")]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 3
        assert "tool_result" in builder.to_jsonl()

    def test_tool_role_without_tool_call_id_skipped(self):
        """Tool messages without tool_call_id are silently skipped."""
        builder = TranscriptBuilder()
        msgs = [ChatMessage(role="tool", tool_call_id=None, content="orphan")]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 0

    def test_tool_call_missing_function_key_uses_unknown_name(self):
        """A tool_call dict with no 'function' key uses 'unknown' as the tool name."""
        builder = TranscriptBuilder()
        # Tool call dict exists but 'function' sub-dict is missing entirely
        msgs = [
            ChatMessage(role="assistant", content=None, tool_calls=[{"id": "tc-x"}])
        ]
        _append_gap_to_builder(msgs, builder)
        assert builder.entry_count == 1
        jsonl = builder.to_jsonl()
        assert "unknown" in jsonl
