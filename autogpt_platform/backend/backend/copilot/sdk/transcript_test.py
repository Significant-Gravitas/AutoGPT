"""Unit tests for JSONL transcript management utilities."""

import json
import os

from .transcript import (
    STRIPPABLE_TYPES,
    read_transcript_file,
    strip_progress_entries,
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
PROGRESS_ENTRY = {
    "type": "progress",
    "uuid": "p1",
    "parentUuid": "u1",
    "data": {"type": "bash_progress", "stdout": "running..."},
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

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n{}\n{}\n")
        assert read_transcript_file(str(path)) is None

    def test_no_size_limit(self, tmp_path):
        """Large files are accepted — bucket storage has no size limit."""
        big_content = {"type": "user", "uuid": "u9", "data": "x" * 1_000_000}
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY, big_content, ASST_MSG)
        path = tmp_path / "big.jsonl"
        path.write_text(content)
        result = read_transcript_file(str(path))
        assert result is not None


# --- write_transcript_to_tempfile ---


class TestWriteTranscriptToTempfile:
    """Tests use /tmp/copilot-* paths to satisfy the sandbox prefix check."""

    def test_writes_file_and_returns_path(self):
        cwd = "/tmp/copilot-test-write"
        try:
            result = write_transcript_to_tempfile(
                VALID_TRANSCRIPT, "sess-1234-abcd", cwd
            )
            assert result is not None
            assert os.path.isfile(result)
            assert result.endswith(".jsonl")
            with open(result) as f:
                assert f.read() == VALID_TRANSCRIPT
        finally:
            import shutil

            shutil.rmtree(cwd, ignore_errors=True)

    def test_creates_parent_directory(self):
        cwd = "/tmp/copilot-test-mkdir"
        try:
            result = write_transcript_to_tempfile(VALID_TRANSCRIPT, "sess-1234", cwd)
            assert result is not None
            assert os.path.isdir(cwd)
        finally:
            import shutil

            shutil.rmtree(cwd, ignore_errors=True)

    def test_uses_session_id_prefix(self):
        cwd = "/tmp/copilot-test-prefix"
        try:
            result = write_transcript_to_tempfile(
                VALID_TRANSCRIPT, "abcdef12-rest", cwd
            )
            assert result is not None
            assert "abcdef12" in os.path.basename(result)
        finally:
            import shutil

            shutil.rmtree(cwd, ignore_errors=True)

    def test_rejects_cwd_outside_sandbox(self, tmp_path):
        cwd = str(tmp_path / "not-copilot")
        result = write_transcript_to_tempfile(VALID_TRANSCRIPT, "sess-1234", cwd)
        assert result is None


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


# --- strip_progress_entries ---


class TestStripProgressEntries:
    def test_strips_all_strippable_types(self):
        """All STRIPPABLE_TYPES are removed from the output."""
        entries = [
            USER_MSG,
            {"type": "progress", "uuid": "p1", "parentUuid": "u1"},
            {"type": "file-history-snapshot", "files": []},
            {"type": "queue-operation", "subtype": "create"},
            {"type": "summary", "text": "..."},
            {"type": "pr-link", "url": "..."},
            ASST_MSG,
        ]
        result = strip_progress_entries(_make_jsonl(*entries))
        result_types = {json.loads(line)["type"] for line in result.strip().split("\n")}
        assert result_types == {"user", "assistant"}
        for stype in STRIPPABLE_TYPES:
            assert stype not in result_types

    def test_reparents_children_of_stripped_entries(self):
        """An assistant message whose parent is a progress entry gets reparented."""
        progress = {
            "type": "progress",
            "uuid": "p1",
            "parentUuid": "u1",
            "data": {"type": "bash_progress"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "p1",  # Points to progress
            "message": {"role": "assistant", "content": "done"},
        }
        content = _make_jsonl(USER_MSG, progress, asst)
        result = strip_progress_entries(content)
        lines = [json.loads(line) for line in result.strip().split("\n")]

        asst_entry = next(e for e in lines if e["type"] == "assistant")
        # Should be reparented to u1 (the user message)
        assert asst_entry["parentUuid"] == "u1"

    def test_reparents_through_chain(self):
        """Reparenting walks through multiple stripped entries."""
        p1 = {"type": "progress", "uuid": "p1", "parentUuid": "u1"}
        p2 = {"type": "progress", "uuid": "p2", "parentUuid": "p1"}
        p3 = {"type": "progress", "uuid": "p3", "parentUuid": "p2"}
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "p3",  # 3 levels deep
            "message": {"role": "assistant", "content": "done"},
        }
        content = _make_jsonl(USER_MSG, p1, p2, p3, asst)
        result = strip_progress_entries(content)
        lines = [json.loads(line) for line in result.strip().split("\n")]

        asst_entry = next(e for e in lines if e["type"] == "assistant")
        assert asst_entry["parentUuid"] == "u1"

    def test_preserves_non_strippable_entries(self):
        """User, assistant, and system entries are preserved."""
        system = {"type": "system", "uuid": "s1", "message": "prompt"}
        content = _make_jsonl(system, USER_MSG, ASST_MSG)
        result = strip_progress_entries(content)
        result_types = [json.loads(line)["type"] for line in result.strip().split("\n")]
        assert result_types == ["system", "user", "assistant"]

    def test_empty_input(self):
        result = strip_progress_entries("")
        # Should return just a newline (empty content stripped)
        assert result.strip() == ""

    def test_no_strippable_entries(self):
        """When there's nothing to strip, output matches input structure."""
        content = _make_jsonl(USER_MSG, ASST_MSG)
        result = strip_progress_entries(content)
        result_lines = result.strip().split("\n")
        assert len(result_lines) == 2

    def test_handles_entries_without_uuid(self):
        """Entries without uuid field are handled gracefully."""
        no_uuid = {"type": "queue-operation", "subtype": "create"}
        content = _make_jsonl(no_uuid, USER_MSG, ASST_MSG)
        result = strip_progress_entries(content)
        result_types = [json.loads(line)["type"] for line in result.strip().split("\n")]
        # queue-operation is strippable
        assert "queue-operation" not in result_types
        assert "user" in result_types
        assert "assistant" in result_types
