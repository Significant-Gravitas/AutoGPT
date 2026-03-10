"""Unit tests for JSONL transcript management utilities."""

import os

from backend.util import json

from .transcript import (
    STRIPPABLE_TYPES,
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
        """With --resume the user message is a CLI query param, not a transcript entry.
        A transcript with only assistant entries is valid."""
        content = _make_jsonl(METADATA_LINE, FILE_HISTORY, ASST_MSG)
        assert validate_transcript(content) is True

    def test_resume_transcript_without_user_entry(self):
        """Simulates a real --resume stop hook transcript: the CLI session file
        has summary + assistant entries but no user entry."""
        summary = {"type": "summary", "uuid": "s1", "text": "context..."}
        asst1 = {
            "type": "assistant",
            "uuid": "a1",
            "message": {"role": "assistant", "content": "Hello!"},
        }
        asst2 = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "a1",
            "message": {"role": "assistant", "content": "Sure, let me help."},
        }
        content = _make_jsonl(summary, asst1, asst2)
        assert validate_transcript(content) is True

    def test_single_assistant_entry(self):
        """A transcript with just one assistant line is valid — the CLI may
        produce short transcripts for simple responses with no tool use."""
        content = json.dumps(ASST_MSG) + "\n"
        assert validate_transcript(content) is True

    def test_invalid_json_returns_false(self):
        assert validate_transcript("not json\n{}\n{}\n") is False

    def test_malformed_json_after_valid_assistant_returns_false(self):
        """Validation must scan all lines - malformed JSON anywhere should fail."""
        valid_asst = json.dumps(ASST_MSG)
        malformed = "not valid json"
        content = valid_asst + "\n" + malformed + "\n"
        assert validate_transcript(content) is False

    def test_blank_lines_are_skipped(self):
        """Transcripts with blank lines should be valid if they contain assistant entries."""
        content = (
            json.dumps(USER_MSG)
            + "\n\n"  # blank line
            + json.dumps(ASST_MSG)
            + "\n"
            + "\n"  # another blank line
        )
        assert validate_transcript(content) is True


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

    def test_preserves_original_line_formatting(self):
        """Non-reparented entries keep their original JSON formatting."""
        # orjson produces compact JSON - test that we preserve the exact input
        # when no reparenting is needed (no re-serialization)
        original_line = json.dumps(USER_MSG)

        content = original_line + "\n" + json.dumps(ASST_MSG) + "\n"
        result = strip_progress_entries(content)
        result_lines = result.strip().split("\n")

        # Original line should be byte-identical (not re-serialized)
        assert result_lines[0] == original_line

    def test_reparented_entries_are_reserialized(self):
        """Entries whose parentUuid changes must be re-serialized."""
        progress = {"type": "progress", "uuid": "p1", "parentUuid": "u1"}
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "p1",
            "message": {"role": "assistant", "content": "done"},
        }
        content = _make_jsonl(USER_MSG, progress, asst)
        result = strip_progress_entries(content)
        lines = result.strip().split("\n")
        asst_entry = json.loads(lines[-1])
        assert asst_entry["parentUuid"] == "u1"  # reparented
