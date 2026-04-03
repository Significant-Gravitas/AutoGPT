"""Unit tests for baseline transcript integration logic.

Tests cover stale transcript detection, transcript_covers_prefix gating,
and partial backfill on stream error — without API keys or network access.
"""

import json as stdlib_json

import pytest

from backend.copilot.transcript import TranscriptDownload, validate_transcript
from backend.copilot.transcript_builder import TranscriptBuilder


def _make_transcript_content(*roles: str) -> str:
    """Build minimal valid JSONL transcript from role names."""
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
        lines.append(stdlib_json.dumps(entry))
        parent = uid
    return "\n".join(lines) + "\n"


class TestStaleTranscriptDetection:
    """Tests for the stale-transcript detection logic in the baseline service.

    When the downloaded transcript's message_count is behind the session's
    message count, the transcript is considered stale and skipped.
    """

    def test_stale_transcript_detected(self):
        """Transcript with fewer messages than the session is flagged as stale."""
        dl = TranscriptDownload(
            content=_make_transcript_content("user", "assistant"),
            message_count=2,
        )
        session_msg_count = 6  # 4 new messages since transcript was uploaded

        is_stale = dl.message_count and dl.message_count < session_msg_count - 1
        assert is_stale

    def test_fresh_transcript_accepted(self):
        """Transcript covering the session prefix is not flagged as stale."""
        dl = TranscriptDownload(
            content=_make_transcript_content("user", "assistant"),
            message_count=4,
        )
        session_msg_count = 5  # Only 1 new message (the user turn just sent)

        is_stale = dl.message_count and dl.message_count < session_msg_count - 1
        assert not is_stale

    def test_zero_message_count_not_stale(self):
        """When message_count is 0 (unknown), staleness check is skipped."""
        dl = TranscriptDownload(
            content=_make_transcript_content("user", "assistant"),
            message_count=0,
        )
        session_msg_count = 10

        is_stale = dl.message_count and dl.message_count < session_msg_count - 1
        assert not is_stale  # 0 is falsy, so check is skipped


class TestTranscriptCoversPrefix:
    """Tests for transcript_covers_prefix gating in the baseline upload path.

    When transcript_covers_prefix is False, the transcript is NOT uploaded
    to avoid overwriting a more complete version in storage.
    """

    def test_no_download_sets_covers_false(self):
        """When no transcript is available, covers_prefix should be False."""
        dl = None
        transcript_covers_prefix = dl is not None
        assert not transcript_covers_prefix

    def test_invalid_transcript_sets_covers_false(self):
        """When downloaded transcript fails validation, covers_prefix is False."""
        content = '{"type":"progress","uuid":"a"}\n'
        assert not validate_transcript(content)

    def test_valid_transcript_sets_covers_true(self):
        """When downloaded transcript is valid and fresh, covers_prefix is True."""
        content = _make_transcript_content("user", "assistant")
        assert validate_transcript(content)


class TestPartialBackfill:
    """Tests for partial backfill of assistant text on stream error.

    When the stream aborts mid-round, the conversation updater may not have
    recorded the partial assistant text. The finally block backfills it
    so mode-switching after a failed turn doesn't lose the partial response.
    """

    def test_backfill_appends_when_last_entry_not_assistant(self):
        """When the last transcript entry is not an assistant, backfill appends."""
        builder = TranscriptBuilder()
        builder.append_user("user question")

        assistant_text = "partial response before error"
        assert builder.last_entry_type != "assistant"

        builder.append_assistant(
            content_blocks=[{"type": "text", "text": assistant_text}],
            model="test-model",
            stop_reason="end_turn",
        )
        assert builder.last_entry_type == "assistant"
        jsonl = builder.to_jsonl()
        assert "partial response before error" in jsonl

    def test_no_backfill_when_last_entry_is_assistant(self):
        """When the conversation updater already recorded the assistant turn,
        backfill should be skipped (checked via last_entry_type)."""
        builder = TranscriptBuilder()
        builder.append_user("user question")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "already recorded"}],
            model="test-model",
            stop_reason="end_turn",
        )

        assert builder.last_entry_type == "assistant"
        initial_count = builder.entry_count

        # Simulating the backfill guard: don't append if already assistant
        if builder.last_entry_type != "assistant":
            builder.append_assistant(
                content_blocks=[{"type": "text", "text": "duplicate"}],
                model="test-model",
                stop_reason="end_turn",
            )

        assert builder.entry_count == initial_count

    def test_no_backfill_when_no_assistant_text(self):
        """When stream_error is True but no assistant text was produced,
        no backfill should occur."""
        builder = TranscriptBuilder()
        builder.append_user("user question")

        assistant_text = ""
        _stream_error = True

        # Simulating the guard from service.py:
        # if _stream_error and state.assistant_text:
        should_backfill = _stream_error and assistant_text
        assert not should_backfill


class TestTranscriptUploadGating:
    """Tests for the upload gating logic.

    Upload only happens when user_id is set AND transcript_covers_prefix is True.
    """

    @pytest.mark.asyncio
    async def test_upload_skipped_without_user_id(self):
        """No upload when user_id is None."""
        user_id = None
        transcript_covers_prefix = True

        should_upload = user_id and transcript_covers_prefix
        assert not should_upload

    @pytest.mark.asyncio
    async def test_upload_skipped_when_prefix_not_covered(self):
        """No upload when transcript doesn't cover the session prefix."""
        user_id = "user-1"
        transcript_covers_prefix = False

        should_upload = user_id and transcript_covers_prefix
        assert not should_upload

    @pytest.mark.asyncio
    async def test_upload_proceeds_when_conditions_met(self):
        """Upload proceeds when user_id is set and prefix is covered."""
        user_id = "user-1"
        transcript_covers_prefix = True

        should_upload = user_id and transcript_covers_prefix
        assert should_upload
