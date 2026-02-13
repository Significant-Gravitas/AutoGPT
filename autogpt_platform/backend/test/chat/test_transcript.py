"""Unit tests for JSONL transcript management utilities."""

import json
import os

from backend.api.features.chat.sdk.transcript import (
    MAX_TRANSCRIPT_SIZE,
    read_transcript_file,
    validate_transcript,
    write_transcript_to_tempfile,
)


def _make_jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


# --- Fixtures ---


METADATA_LINE = {"type": "queue-operation", "subtype": "create"}
FILE_HISTORY = {"type": "file-history-snapshot", "files": []}
USER_MSG = {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "hi"}}
ASST_MSG = {
    "type": "assistant",
    "uuid": "a1",
    "parentUuid": "u1",
    "message": {"role": "assistant", "content": "hello"},
}

VALID_TRANSCRIPT = _make_jsonl(METADATA_LINE, FILE_HISTORY, USER_MSG, ASST_MSG)


# --- read_transcript_file ---


class TestReadTranscriptFile:
    def test_returns_content_for_valid_file(self, tmp_path):
        path = tmp_path / "session.jsonl"
        path.write_text(VALID_TRANSCRIPT)
        result = read_transcript_file(str(path))
        assert result is not None
        assert "user" in result

    def test_returns_none_for_missing_file(self):
        assert read_transcript_file("/nonexistent/path.jsonl") is None

    def test_returns_none_for_empty_path(self):
        assert read_transcript_file("") is None

    def test_returns_none_for_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert read_transcript_file(str(path)) is None

    def test_returns_none_for_metadata_only(self, tmp_path):
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY)
        path = tmp_path / "meta.jsonl"
        path.write_text(content)
        assert read_transcript_file(str(path)) is None

    def test_returns_none_for_oversized_file(self, tmp_path):
        # Build a valid transcript that exceeds MAX_TRANSCRIPT_SIZE
        big_content = {"type": "user", "data": "x" * (MAX_TRANSCRIPT_SIZE + 100)}
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY, big_content, ASST_MSG)
        path = tmp_path / "big.jsonl"
        path.write_text(content)
        assert read_transcript_file(str(path)) is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n{}\n{}\n")
        assert read_transcript_file(str(path)) is None


# --- write_transcript_to_tempfile ---


class TestWriteTranscriptToTempfile:
    def test_writes_file_and_returns_path(self, tmp_path):
        cwd = str(tmp_path / "workspace")
        result = write_transcript_to_tempfile(VALID_TRANSCRIPT, "sess-1234-abcd", cwd)
        assert result is not None
        assert os.path.isfile(result)
        assert result.endswith(".jsonl")
        with open(result) as f:
            assert f.read() == VALID_TRANSCRIPT

    def test_creates_parent_directory(self, tmp_path):
        cwd = str(tmp_path / "new" / "dir")
        result = write_transcript_to_tempfile(VALID_TRANSCRIPT, "sess-1234", cwd)
        assert result is not None
        assert os.path.isdir(cwd)

    def test_uses_session_id_prefix(self, tmp_path):
        cwd = str(tmp_path)
        result = write_transcript_to_tempfile(VALID_TRANSCRIPT, "abcdef12-rest", cwd)
        assert result is not None
        assert "abcdef12" in os.path.basename(result)


# --- validate_transcript ---


class TestValidateTranscript:
    def test_valid_transcript(self):
        assert validate_transcript(VALID_TRANSCRIPT) is True

    def test_none_content(self):
        assert validate_transcript(None) is False

    def test_empty_content(self):
        assert validate_transcript("") is False

    def test_metadata_only(self):
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY)
        assert validate_transcript(content) is False

    def test_user_only_no_assistant(self):
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY, USER_MSG)
        assert validate_transcript(content) is False

    def test_assistant_only_no_user(self):
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY, ASST_MSG)
        assert validate_transcript(content) is False

    def test_invalid_json_returns_false(self):
        assert validate_transcript("not json\n{}\n{}\n") is False
