"""Unit tests for JSONL transcript management utilities."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util import json

from .transcript import (
    STRIPPABLE_TYPES,
    delete_transcript,
    read_compacted_entries,
    strip_progress_entries,
    strip_stale_thinking_blocks,
    validate_transcript,
    write_transcript_to_tempfile,
)
from .transcript_builder import TranscriptBuilder


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


# --- delete_transcript ---


class TestDeleteTranscript:
    @pytest.mark.asyncio
    async def test_deletes_cli_session_and_meta(self):
        """delete_transcript removes the CLI session .jsonl and .meta.json."""
        mock_storage = AsyncMock()
        mock_storage.delete = AsyncMock()

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            await delete_transcript("user-123", "session-456")

        assert mock_storage.delete.call_count == 2
        paths = [call.args[0] for call in mock_storage.delete.call_args_list]
        assert any(p.endswith(".jsonl") for p in paths)
        assert any(p.endswith(".meta.json") for p in paths)

    @pytest.mark.asyncio
    async def test_continues_on_jsonl_delete_failure(self):
        """If .jsonl delete fails, .meta.json delete is still attempted."""
        mock_storage = AsyncMock()
        mock_storage.delete = AsyncMock(
            side_effect=[Exception("jsonl delete failed"), None]
        )

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            # Should not raise
            await delete_transcript("user-123", "session-456")

        assert mock_storage.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_meta_delete_failure(self):
        """If .meta.json delete fails, no exception propagates."""
        mock_storage = AsyncMock()
        mock_storage.delete = AsyncMock(
            side_effect=[None, Exception("meta delete failed")]
        )

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            # Should not raise
            await delete_transcript("user-123", "session-456")


# --- read_compacted_entries ---


COMPACT_SUMMARY = {
    "type": "summary",
    "uuid": "cs1",
    "isCompactSummary": True,
    "message": {"role": "assistant", "content": "compacted context"},
}
POST_COMPACT_ASST = {
    "type": "assistant",
    "uuid": "a2",
    "parentUuid": "cs1",
    "message": {"role": "assistant", "content": "response after compaction"},
}


class TestReadCompactedEntries:
    def test_returns_summary_and_entries_after(self, tmp_path, monkeypatch):
        """File with isCompactSummary entry returns summary + entries after."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        pre_compact = {"type": "user", "uuid": "u1", "message": {"role": "user"}}
        path = session_dir / "session.jsonl"
        path.write_text(_make_jsonl(pre_compact, COMPACT_SUMMARY, POST_COMPACT_ASST))

        result = read_compacted_entries(str(path))
        assert result is not None
        assert len(result) == 2
        assert result[0]["isCompactSummary"] is True
        assert result[1]["uuid"] == "a2"

    def test_no_compact_summary_returns_none(self, tmp_path, monkeypatch):
        """File without isCompactSummary returns None."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        path = session_dir / "session.jsonl"
        path.write_text(_make_jsonl(USER_MSG, ASST_MSG))

        result = read_compacted_entries(str(path))
        assert result is None

    def test_file_not_found_returns_none(self, tmp_path, monkeypatch):
        """Non-existent file returns None."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        projects_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        result = read_compacted_entries(str(projects_dir / "missing.jsonl"))
        assert result is None

    def test_empty_path_returns_none(self):
        """Empty string path returns None."""
        result = read_compacted_entries("")
        assert result is None

    def test_malformed_json_lines_skipped(self, tmp_path, monkeypatch):
        """Malformed JSON lines are skipped gracefully."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        path = session_dir / "session.jsonl"
        content = "not valid json\n" + json.dumps(COMPACT_SUMMARY) + "\n"
        content += "also bad\n" + json.dumps(POST_COMPACT_ASST) + "\n"
        path.write_text(content)

        result = read_compacted_entries(str(path))
        assert result is not None
        assert len(result) == 2  # summary + post-compact assistant

    def test_multiple_compact_summaries_uses_last(self, tmp_path, monkeypatch):
        """When multiple isCompactSummary entries exist, uses the last one
        (most recent compaction)."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        second_summary = {
            "type": "summary",
            "uuid": "cs2",
            "isCompactSummary": True,
            "message": {"role": "assistant", "content": "second summary"},
        }
        path = session_dir / "session.jsonl"
        path.write_text(_make_jsonl(COMPACT_SUMMARY, POST_COMPACT_ASST, second_summary))

        result = read_compacted_entries(str(path))
        assert result is not None
        # Last summary found, so only cs2 returned
        assert len(result) == 1
        assert result[0]["uuid"] == "cs2"

    def test_path_outside_projects_base_returns_none(self, tmp_path, monkeypatch):
        """Transcript path outside the projects directory is rejected."""
        config_dir = tmp_path / "config"
        (config_dir / "projects").mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        evil_file = tmp_path / "evil.jsonl"
        evil_file.write_text(_make_jsonl(COMPACT_SUMMARY))

        result = read_compacted_entries(str(evil_file))
        assert result is None


# --- TranscriptBuilder.replace_entries ---


class TestTranscriptBuilderReplaceEntries:
    def test_replaces_existing_entries(self):
        """replace_entries replaces all entries with compacted ones."""
        builder = TranscriptBuilder()
        builder.append_user("hello")
        builder.append_assistant([{"type": "text", "text": "world"}])
        assert builder.entry_count == 2

        compacted = [
            {
                "type": "user",
                "uuid": "cs1",
                "isCompactSummary": True,
                "message": {"role": "user", "content": "compacted summary"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "cs1",
                "message": {"role": "assistant", "content": "response"},
            },
        ]
        builder.replace_entries(compacted)
        assert builder.entry_count == 2
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[0]["uuid"] == "cs1"
        assert entries[1]["uuid"] == "a1"

    def test_filters_strippable_types(self):
        """Strippable types are filtered out during replace."""
        builder = TranscriptBuilder()
        compacted = [
            {
                "type": "user",
                "uuid": "cs1",
                "message": {"role": "user", "content": "compacted summary"},
            },
            {"type": "progress", "uuid": "p1", "message": {}},
            {"type": "summary", "uuid": "s1", "message": {}},
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "cs1",
                "message": {"role": "assistant", "content": "hi"},
            },
        ]
        builder.replace_entries(compacted)
        assert builder.entry_count == 2  # progress and summary were filtered

    def test_maintains_last_uuid_chain(self):
        """After replace, _last_uuid is the last entry's uuid."""
        builder = TranscriptBuilder()
        compacted = [
            {
                "type": "user",
                "uuid": "cs1",
                "message": {"role": "user", "content": "compacted summary"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "cs1",
                "message": {"role": "assistant", "content": "hi"},
            },
        ]
        builder.replace_entries(compacted)
        # Appending a new user message should chain to a1
        builder.append_user("next question")
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[-1]["parentUuid"] == "a1"

    def test_empty_entries_list_keeps_existing(self):
        """Replacing with empty list keeps existing entries (safety check)."""
        builder = TranscriptBuilder()
        builder.append_user("hello")
        builder.replace_entries([])
        # Empty input is treated as corrupt — existing entries preserved
        assert builder.entry_count == 1
        assert not builder.is_empty


# --- TranscriptBuilder.load_previous with compacted content ---


class TestTranscriptBuilderLoadPreviousCompacted:
    def test_preserves_compact_summary_entry(self):
        """load_previous preserves isCompactSummary entries even though
        their type is 'summary' (which is in STRIPPABLE_TYPES)."""
        compacted_content = _make_jsonl(COMPACT_SUMMARY, POST_COMPACT_ASST)
        builder = TranscriptBuilder()
        builder.load_previous(compacted_content)
        assert builder.entry_count == 2
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[0]["type"] == "summary"
        assert entries[0]["uuid"] == "cs1"
        assert entries[1]["uuid"] == "a2"

    def test_strips_regular_summary_entries(self):
        """Regular summary entries (without isCompactSummary) are still stripped."""
        regular_summary = {"type": "summary", "uuid": "s1", "message": {"content": "x"}}
        content = _make_jsonl(regular_summary, POST_COMPACT_ASST)
        builder = TranscriptBuilder()
        builder.load_previous(content)
        assert builder.entry_count == 1  # Only the assistant entry


# --- End-to-end compaction flow (simulates service.py) ---


class TestCompactionFlowIntegration:
    """Simulate the full compaction flow as it happens in service.py:

    1. TranscriptBuilder loads a previous transcript (download)
    2. New messages are appended (user query + assistant response)
    3. CompactionTracker fires (PreCompact hook → emit_start → emit_end)
    4. read_compacted_entries reads the CLI session file
    5. TranscriptBuilder.replace_entries syncs with CLI state
    6. Final to_jsonl() produces the correct output (upload)
    """

    def test_full_compaction_roundtrip(self, tmp_path, monkeypatch):
        """Full roundtrip: load → append → compact → replace → export."""
        # Setup: create a CLI session file with pre-compact + compaction entries
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        # Simulate a transcript with old messages, then a compaction summary
        old_user = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "old question"},
        }
        old_asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {"role": "assistant", "content": "old answer"},
        }
        compact_summary = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "compacted summary of conversation"},
        }
        post_compact_asst = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "cs1",
            "message": {"role": "assistant", "content": "response after compaction"},
        }
        session_file = session_dir / "session.jsonl"
        session_file.write_text(
            _make_jsonl(old_user, old_asst, compact_summary, post_compact_asst)
        )

        # Step 1: TranscriptBuilder loads previous transcript (simulates download)
        # The previous transcript would have the OLD entries (pre-compaction)
        previous_transcript = _make_jsonl(old_user, old_asst)
        builder = TranscriptBuilder()
        builder.load_previous(previous_transcript)
        assert builder.entry_count == 2

        # Step 2: New messages appended during the current query
        builder.append_user("new question")
        builder.append_assistant([{"type": "text", "text": "new answer"}])
        assert builder.entry_count == 4

        # Step 3: read_compacted_entries reads the CLI session file
        compacted = read_compacted_entries(str(session_file))
        assert compacted is not None
        assert len(compacted) == 2  # compact_summary + post_compact_asst
        assert compacted[0]["isCompactSummary"] is True

        # Step 4: replace_entries syncs builder with CLI state
        builder.replace_entries(compacted)
        assert builder.entry_count == 2  # Only compacted entries now

        # Step 5: Append post-compaction messages (continuing the stream)
        builder.append_user("follow-up question")
        assert builder.entry_count == 3

        # Step 6: Export and verify
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert len(entries) == 3
        # First entry is the compaction summary
        assert entries[0]["type"] == "summary"
        assert entries[0]["uuid"] == "cs1"
        # Second is the post-compact assistant
        assert entries[1]["uuid"] == "a2"
        # Third is our follow-up, parented to the last compacted entry
        assert entries[2]["type"] == "user"
        assert entries[2]["parentUuid"] == "a2"

    def test_compaction_preserves_chain_across_multiple_compactions(
        self, tmp_path, monkeypatch
    ):
        """Two compactions: first compacts old history, second compacts the first."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        # First compaction
        first_summary = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "first summary"},
        }
        mid_asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "cs1",
            "message": {"role": "assistant", "content": "mid response"},
        }
        # Second compaction (compacts the first summary + mid_asst)
        second_summary = {
            "type": "summary",
            "uuid": "cs2",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "second summary"},
        }
        final_asst = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "cs2",
            "message": {"role": "assistant", "content": "final response"},
        }

        session_file = session_dir / "session.jsonl"
        session_file.write_text(
            _make_jsonl(first_summary, mid_asst, second_summary, final_asst)
        )

        # read_compacted_entries should find the LAST summary
        compacted = read_compacted_entries(str(session_file))
        assert compacted is not None
        assert len(compacted) == 2  # second_summary + final_asst
        assert compacted[0]["uuid"] == "cs2"

        # Apply to builder
        builder = TranscriptBuilder()
        builder.append_user("old stuff")
        builder.append_assistant([{"type": "text", "text": "old response"}])
        builder.replace_entries(compacted)
        assert builder.entry_count == 2

        # New message chains correctly
        builder.append_user("after second compaction")
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[-1]["parentUuid"] == "a2"

    def test_strip_progress_preserves_compact_summaries(self):
        """strip_progress_entries doesn't strip isCompactSummary entries
        even though their type is 'summary' (in STRIPPABLE_TYPES)."""
        compact_summary = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "compacted"},
        }
        regular_summary = {"type": "summary", "uuid": "s1", "message": {"content": "x"}}
        progress = {"type": "progress", "uuid": "p1", "data": {"stdout": "..."}}
        user = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "hi"},
        }

        content = _make_jsonl(compact_summary, regular_summary, progress, user)
        stripped = strip_progress_entries(content)
        stripped_entries = [
            json.loads(line) for line in stripped.strip().split("\n") if line.strip()
        ]

        uuids = [e.get("uuid") for e in stripped_entries]
        # compact_summary kept, regular_summary stripped, progress stripped, user kept
        assert "cs1" in uuids  # compact summary preserved
        assert "s1" not in uuids  # regular summary stripped
        assert "p1" not in uuids  # progress stripped
        assert "u1" in uuids  # user kept

    def test_builder_load_then_replace_then_export_roundtrip(self):
        """Load a compacted transcript, replace with new compaction, export.
        Simulates two consecutive turns with compaction each time."""
        # Turn 1: load compacted transcript
        compact1 = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "summary v1"},
        }
        asst1 = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "cs1",
            "message": {"role": "assistant", "content": "response 1"},
        }
        builder = TranscriptBuilder()
        builder.load_previous(_make_jsonl(compact1, asst1))
        assert builder.entry_count == 2

        # Turn 1: append new messages
        builder.append_user("question")
        builder.append_assistant([{"type": "text", "text": "answer"}])
        assert builder.entry_count == 4

        # Turn 1: compaction fires — replace with new compacted state
        compact2 = {
            "type": "summary",
            "uuid": "cs2",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "summary v2"},
        }
        asst2 = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "cs2",
            "message": {"role": "assistant", "content": "continuing"},
        }
        builder.replace_entries([compact2, asst2])
        assert builder.entry_count == 2

        # Export (this goes to cloud storage for next turn's download)
        output = builder.to_jsonl()
        lines = [json.loads(line) for line in output.strip().split("\n")]
        assert lines[0]["uuid"] == "cs2"
        assert lines[0]["type"] == "summary"
        assert lines[1]["uuid"] == "a2"

        # Turn 2: fresh builder loads the exported transcript
        builder2 = TranscriptBuilder()
        builder2.load_previous(output)
        assert builder2.entry_count == 2
        builder2.append_user("turn 2 question")
        output2 = builder2.to_jsonl()
        lines2 = [json.loads(line) for line in output2.strip().split("\n")]
        assert lines2[-1]["parentUuid"] == "a2"


# ---------------------------------------------------------------------------
# _run_compression (direct tests for the 3 code paths)
# ---------------------------------------------------------------------------


class TestRunCompression:
    """Direct tests for ``_run_compression`` covering all 3 code paths.

    Paths:
    (a) No OpenAI client configured → truncation fallback immediately.
    (b) LLM success → returns LLM-compressed result.
    (c) LLM call raises → truncation fallback.
    """

    def _make_compress_result(self, was_compacted: bool, msgs=None):
        """Build a minimal CompressResult-like object."""
        from types import SimpleNamespace

        return SimpleNamespace(
            was_compacted=was_compacted,
            messages=msgs or [{"role": "user", "content": "summary"}],
            original_token_count=500,
            token_count=100 if was_compacted else 500,
            messages_summarized=2 if was_compacted else 0,
            messages_dropped=0,
        )

    @pytest.mark.asyncio
    async def test_no_client_uses_truncation(self):
        """Path (a): ``get_openai_client()`` returns None → truncation only."""
        from backend.copilot.transcript import _run_compression

        truncation_result = self._make_compress_result(
            True, [{"role": "user", "content": "truncated"}]
        )

        with (
            patch(
                "backend.copilot.transcript.get_openai_client",
                return_value=None,
            ),
            patch(
                "backend.copilot.transcript.compress_context",
                new_callable=AsyncMock,
                return_value=truncation_result,
            ) as mock_compress,
        ):
            result = await _run_compression(
                [{"role": "user", "content": "hello"}],
                model="test-model",
                log_prefix="[test]",
            )

        # compress_context called with client=None (truncation mode)
        call_kwargs = mock_compress.call_args
        assert (
            call_kwargs.kwargs.get("client") is None
            or (call_kwargs.args and call_kwargs.args[2] is None)
            or mock_compress.call_args[1].get("client") is None
        )
        assert result is truncation_result

    @pytest.mark.asyncio
    async def test_llm_success_returns_llm_result(self):
        """Path (b): ``get_openai_client()`` returns a client → LLM compresses."""
        from backend.copilot.transcript import _run_compression

        llm_result = self._make_compress_result(
            True, [{"role": "user", "content": "LLM summary"}]
        )
        mock_client = MagicMock()

        with (
            patch(
                "backend.copilot.transcript.get_openai_client",
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.transcript.compress_context",
                new_callable=AsyncMock,
                return_value=llm_result,
            ) as mock_compress,
        ):
            result = await _run_compression(
                [{"role": "user", "content": "long conversation"}],
                model="test-model",
                log_prefix="[test]",
            )

        # compress_context called with the real client
        assert mock_compress.called
        assert result is llm_result

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_truncation(self):
        """Path (c): LLM call raises → truncation fallback used instead."""
        from backend.copilot.transcript import _run_compression

        truncation_result = self._make_compress_result(
            True, [{"role": "user", "content": "truncated fallback"}]
        )
        mock_client = MagicMock()
        call_count = [0]

        async def _compress_side_effect(**kwargs):
            call_count[0] += 1
            if kwargs.get("client") is not None:
                raise RuntimeError("LLM timeout")
            return truncation_result

        with (
            patch(
                "backend.copilot.transcript.get_openai_client",
                return_value=mock_client,
            ),
            patch(
                "backend.copilot.transcript.compress_context",
                side_effect=_compress_side_effect,
            ),
        ):
            result = await _run_compression(
                [{"role": "user", "content": "long conversation"}],
                model="test-model",
                log_prefix="[test]",
            )

        # compress_context called twice: once for LLM (raises), once for truncation
        assert call_count[0] == 2
        assert result is truncation_result

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_to_truncation(self):
        """Path (d): LLM call exceeds timeout → truncation fallback used."""
        from backend.copilot.transcript import _run_compression

        truncation_result = self._make_compress_result(
            True, [{"role": "user", "content": "truncated after timeout"}]
        )
        call_count = [0]

        async def _compress_side_effect(*, messages, model, client, target_tokens=None):
            call_count[0] += 1
            if client is not None:
                # Simulate a hang that exceeds the timeout
                await asyncio.sleep(9999)
            return truncation_result

        fake_client = MagicMock()
        with (
            patch(
                "backend.copilot.transcript.get_openai_client",
                return_value=fake_client,
            ),
            patch(
                "backend.copilot.transcript.compress_context",
                side_effect=_compress_side_effect,
            ),
            patch(
                "backend.copilot.transcript._COMPACTION_TIMEOUT_SECONDS",
                0.05,
            ),
            patch(
                "backend.copilot.transcript._TRUNCATION_TIMEOUT_SECONDS",
                5,
            ),
        ):
            result = await _run_compression(
                [{"role": "user", "content": "long conversation"}],
                model="test-model",
                log_prefix="[test]",
            )

        # compress_context called twice: once for LLM (times out), once truncation
        assert call_count[0] == 2
        assert result is truncation_result


# ---------------------------------------------------------------------------
# cleanup_stale_project_dirs
# ---------------------------------------------------------------------------


class TestCleanupStaleProjectDirs:
    """Tests for cleanup_stale_project_dirs (disk leak prevention)."""

    def test_removes_old_copilot_dirs(self, tmp_path, monkeypatch):
        """Directories matching copilot pattern older than threshold are removed."""
        from backend.copilot.transcript import (
            _STALE_PROJECT_DIR_SECONDS,
            cleanup_stale_project_dirs,
        )

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        # Create a stale dir
        stale = projects_dir / "-tmp-copilot-old-session"
        stale.mkdir()
        # Set mtime to past the threshold
        import time

        old_time = time.time() - _STALE_PROJECT_DIR_SECONDS - 100
        os.utime(stale, (old_time, old_time))

        # Create a fresh dir
        fresh = projects_dir / "-tmp-copilot-new-session"
        fresh.mkdir()

        removed = cleanup_stale_project_dirs()
        assert removed == 1
        assert not stale.exists()
        assert fresh.exists()

    def test_ignores_non_copilot_dirs(self, tmp_path, monkeypatch):
        """Directories not matching copilot pattern are left alone."""
        from backend.copilot.transcript import cleanup_stale_project_dirs

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        # Non-copilot dir that's old
        import time

        other = projects_dir / "some-other-project"
        other.mkdir()
        old_time = time.time() - 999999
        os.utime(other, (old_time, old_time))

        removed = cleanup_stale_project_dirs()
        assert removed == 0
        assert other.exists()

    def test_ttl_boundary_not_removed(self, tmp_path, monkeypatch):
        """A directory exactly at the TTL boundary should NOT be removed."""
        from backend.copilot.transcript import (
            _STALE_PROJECT_DIR_SECONDS,
            cleanup_stale_project_dirs,
        )

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        import time

        # Dir that's exactly at the TTL (age == threshold, not >) — should survive
        boundary = projects_dir / "-tmp-copilot-boundary"
        boundary.mkdir()
        boundary_time = time.time() - _STALE_PROJECT_DIR_SECONDS + 1
        os.utime(boundary, (boundary_time, boundary_time))

        removed = cleanup_stale_project_dirs()
        assert removed == 0
        assert boundary.exists()

    def test_skips_non_directory_entries(self, tmp_path, monkeypatch):
        """Regular files matching the copilot pattern are not removed."""
        from backend.copilot.transcript import (
            _STALE_PROJECT_DIR_SECONDS,
            cleanup_stale_project_dirs,
        )

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        import time

        # Create a regular FILE (not a dir) with the copilot pattern name
        stale_file = projects_dir / "-tmp-copilot-stale-file"
        stale_file.write_text("not a dir")
        old_time = time.time() - _STALE_PROJECT_DIR_SECONDS - 100
        os.utime(stale_file, (old_time, old_time))

        removed = cleanup_stale_project_dirs()
        assert removed == 0
        assert stale_file.exists()

    def test_missing_base_dir_returns_zero(self, tmp_path, monkeypatch):
        """If the projects base directory doesn't exist, return 0 gracefully."""
        from backend.copilot.transcript import cleanup_stale_project_dirs

        nonexistent = str(tmp_path / "does-not-exist" / "projects")
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: nonexistent,
        )

        removed = cleanup_stale_project_dirs()
        assert removed == 0

    def test_scoped_removes_only_target_dir(self, tmp_path, monkeypatch):
        """When encoded_cwd is supplied only that directory is swept."""
        import time

        from backend.copilot.transcript import (
            _STALE_PROJECT_DIR_SECONDS,
            cleanup_stale_project_dirs,
        )

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        old_time = time.time() - _STALE_PROJECT_DIR_SECONDS - 100

        # Two stale copilot dirs
        target = projects_dir / "-tmp-copilot-session-abc"
        target.mkdir()
        os.utime(target, (old_time, old_time))

        other = projects_dir / "-tmp-copilot-session-xyz"
        other.mkdir()
        os.utime(other, (old_time, old_time))

        # Only the target dir should be removed
        removed = cleanup_stale_project_dirs(encoded_cwd="-tmp-copilot-session-abc")
        assert removed == 1
        assert not target.exists()
        assert other.exists()  # untouched — not the current session

    def test_scoped_fresh_dir_not_removed(self, tmp_path, monkeypatch):
        """Scoped sweep leaves a fresh directory alone."""
        from backend.copilot.transcript import cleanup_stale_project_dirs

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        fresh = projects_dir / "-tmp-copilot-session-new"
        fresh.mkdir()
        # mtime is now — well within TTL

        removed = cleanup_stale_project_dirs(encoded_cwd="-tmp-copilot-session-new")
        assert removed == 0
        assert fresh.exists()

    def test_scoped_non_copilot_dir_not_removed(self, tmp_path, monkeypatch):
        """Scoped sweep refuses to remove a non-copilot directory."""
        import time

        from backend.copilot.transcript import (
            _STALE_PROJECT_DIR_SECONDS,
            cleanup_stale_project_dirs,
        )

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        monkeypatch.setattr(
            "backend.copilot.transcript.projects_base",
            lambda: str(projects_dir),
        )

        old_time = time.time() - _STALE_PROJECT_DIR_SECONDS - 100
        non_copilot = projects_dir / "some-other-project"
        non_copilot.mkdir()
        os.utime(non_copilot, (old_time, old_time))

        removed = cleanup_stale_project_dirs(encoded_cwd="some-other-project")
        assert removed == 0
        assert non_copilot.exists()


# ---------------------------------------------------------------------------
# strip_stale_thinking_blocks
# ---------------------------------------------------------------------------


class TestStripStaleThinkingBlocks:
    """Tests for strip_stale_thinking_blocks — removes thinking/redacted_thinking
    blocks from non-last assistant entries to reduce transcript bloat."""

    def _asst_entry(
        self, msg_id: str, content: list, uuid: str = "u1", parent: str = ""
    ) -> dict:
        return {
            "type": "assistant",
            "uuid": uuid,
            "parentUuid": parent,
            "message": {
                "role": "assistant",
                "id": msg_id,
                "type": "message",
                "content": content,
            },
        }

    def _user_entry(self, text: str, uuid: str = "u0", parent: str = "") -> dict:
        return {
            "type": "user",
            "uuid": uuid,
            "parentUuid": parent,
            "message": {"role": "user", "content": text},
        }

    def test_strips_thinking_from_older_assistant(self) -> None:
        """Thinking blocks in non-last assistant entries should be removed."""
        old_asst = self._asst_entry(
            "msg_old",
            [
                {"type": "thinking", "thinking": "deep thoughts..."},
                {"type": "text", "text": "hello"},
                {"type": "redacted_thinking", "data": "secret"},
            ],
            uuid="a1",
        )
        new_asst = self._asst_entry(
            "msg_new",
            [
                {"type": "thinking", "thinking": "latest thoughts"},
                {"type": "text", "text": "world"},
            ],
            uuid="a2",
            parent="a1",
        )
        content = _make_jsonl(old_asst, new_asst)
        result = strip_stale_thinking_blocks(content)
        lines = [json.loads(ln) for ln in result.strip().split("\n")]

        # Old assistant should have thinking blocks stripped
        old_content = lines[0]["message"]["content"]
        assert len(old_content) == 1
        assert old_content[0]["type"] == "text"

        # New (last) assistant should be untouched
        new_content = lines[1]["message"]["content"]
        assert len(new_content) == 2
        assert new_content[0]["type"] == "thinking"
        assert new_content[1]["type"] == "text"

    def test_preserves_last_assistant_thinking(self) -> None:
        """The last assistant entry's thinking blocks must be preserved."""
        entry = self._asst_entry(
            "msg_only",
            [
                {"type": "thinking", "thinking": "must keep"},
                {"type": "text", "text": "response"},
            ],
        )
        content = _make_jsonl(entry)
        result = strip_stale_thinking_blocks(content)
        lines = [json.loads(ln) for ln in result.strip().split("\n")]
        assert len(lines[0]["message"]["content"]) == 2

    def test_no_assistant_entries_returns_unchanged(self) -> None:
        """Transcripts with only user entries should pass through unchanged."""
        user = self._user_entry("hello")
        content = _make_jsonl(user)
        assert strip_stale_thinking_blocks(content) == content

    def test_empty_content_returns_unchanged(self) -> None:
        assert strip_stale_thinking_blocks("") == ""

    def test_multiple_turns_strips_all_but_last(self) -> None:
        """With 3 assistant turns, only the last keeps thinking blocks."""
        entries = [
            self._asst_entry(
                "msg_1",
                [
                    {"type": "thinking", "thinking": "t1"},
                    {"type": "text", "text": "a1"},
                ],
                uuid="a1",
            ),
            self._user_entry("q2", uuid="u2", parent="a1"),
            self._asst_entry(
                "msg_2",
                [
                    {"type": "thinking", "thinking": "t2"},
                    {"type": "text", "text": "a2"},
                ],
                uuid="a2",
                parent="u2",
            ),
            self._user_entry("q3", uuid="u3", parent="a2"),
            self._asst_entry(
                "msg_3",
                [
                    {"type": "thinking", "thinking": "t3"},
                    {"type": "text", "text": "a3"},
                ],
                uuid="a3",
                parent="u3",
            ),
        ]
        content = _make_jsonl(*entries)
        result = strip_stale_thinking_blocks(content)
        lines = [json.loads(ln) for ln in result.strip().split("\n")]

        # msg_1: thinking stripped
        assert len(lines[0]["message"]["content"]) == 1
        assert lines[0]["message"]["content"][0]["type"] == "text"
        # msg_2: thinking stripped
        assert len(lines[2]["message"]["content"]) == 1
        # msg_3 (last): thinking preserved
        assert len(lines[4]["message"]["content"]) == 2
        assert lines[4]["message"]["content"][0]["type"] == "thinking"

    def test_same_msg_id_multi_entry_turn(self) -> None:
        """Multiple entries sharing the same message.id (same turn) are preserved."""
        entries = [
            self._asst_entry(
                "msg_old",
                [{"type": "thinking", "thinking": "old"}],
                uuid="a1",
            ),
            self._asst_entry(
                "msg_last",
                [{"type": "thinking", "thinking": "t_part1"}],
                uuid="a2",
                parent="a1",
            ),
            self._asst_entry(
                "msg_last",
                [{"type": "text", "text": "response"}],
                uuid="a3",
                parent="a2",
            ),
        ]
        content = _make_jsonl(*entries)
        result = strip_stale_thinking_blocks(content)
        lines = [json.loads(ln) for ln in result.strip().split("\n")]

        # Old entry stripped
        assert lines[0]["message"]["content"] == []
        # Both entries of last turn (msg_last) preserved
        assert lines[1]["message"]["content"][0]["type"] == "thinking"
        assert lines[2]["message"]["content"][0]["type"] == "text"


class TestProcessCliRestore:
    """``_process_cli_restore`` validates, strips, and writes CLI session to disk."""

    def test_writes_stripped_bytes_not_raw(self, tmp_path):
        """Stripped bytes (not raw bytes) must be written to disk for --resume."""
        import os
        import re
        from pathlib import Path
        from unittest.mock import patch

        from backend.copilot.sdk.service import _process_cli_restore
        from backend.copilot.transcript import TranscriptDownload

        session_id = "12345678-0000-0000-0000-abcdef000001"
        sdk_cwd = str(tmp_path)
        projects_base_dir = str(tmp_path)

        # Build raw content with a strippable progress entry + a valid user/assistant pair
        raw_content = (
            '{"type":"progress","uuid":"p1","subtype":"agent_progress","parentUuid":null}\n'
            '{"type":"user","uuid":"u1","parentUuid":null,"message":{"role":"user","content":"hi"}}\n'
            '{"type":"assistant","uuid":"a1","parentUuid":"u1","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}\n'
        )
        raw_bytes = raw_content.encode("utf-8")
        restore = TranscriptDownload(content=raw_bytes, message_count=2, mode="sdk")

        with (
            patch(
                "backend.copilot.sdk.service.projects_base",
                return_value=projects_base_dir,
            ),
            patch(
                "backend.copilot.transcript.projects_base",
                return_value=projects_base_dir,
            ),
        ):
            stripped_str, ok = _process_cli_restore(
                restore, sdk_cwd, session_id, "[Test]"
            )

        assert ok, "Expected successful restore"

        # Find the written session file
        encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(sdk_cwd))
        session_file = Path(projects_base_dir) / encoded_cwd / f"{session_id}.jsonl"
        assert session_file.exists(), "Session file should have been written"

        written_bytes = session_file.read_bytes()
        # The written bytes must be the stripped version (no progress entry)
        assert (
            b"progress" not in written_bytes
        ), "Raw bytes with progress entry should not have been written"
        assert (
            b"hello" in written_bytes
        ), "Stripped content should still contain assistant turn"

        # Written bytes must equal the stripped string re-encoded
        assert written_bytes == stripped_str.encode(
            "utf-8"
        ), "Written bytes must equal stripped content"

    def test_invalid_content_returns_false(self):
        """Content that fails validation after strip returns (empty, False)."""
        from backend.copilot.sdk.service import _process_cli_restore
        from backend.copilot.transcript import TranscriptDownload

        # A single progress-only entry — stripped result will be empty/invalid
        raw_content = '{"type":"progress","uuid":"p1","subtype":"agent_progress","parentUuid":null}\n'
        restore = TranscriptDownload(
            content=raw_content.encode("utf-8"), message_count=1, mode="sdk"
        )

        stripped_str, ok = _process_cli_restore(
            restore,
            "/tmp/nonexistent-sdk-cwd",
            "12345678-0000-0000-0000-000000000099",
            "[Test]",
        )

        assert not ok
        assert stripped_str == ""


class TestReadCliSessionFromDisk:
    """``_read_cli_session_from_disk`` reads, strips, and optionally writes back the session."""

    def _build_session_file(self, tmp_path, session_id: str):
        """Build the session file path inside tmp_path using the same encoding as cli_session_path."""
        import os
        import re
        from pathlib import Path

        sdk_cwd = str(tmp_path)
        encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(sdk_cwd))
        session_dir = Path(str(tmp_path)) / encoded_cwd
        session_dir.mkdir(parents=True, exist_ok=True)
        return sdk_cwd, session_dir / f"{session_id}.jsonl"

    def test_returns_raw_bytes_for_invalid_utf8(self, tmp_path):
        """Non-UTF-8 bytes trigger UnicodeDecodeError — returns raw bytes (upload-raw fallback)."""
        from unittest.mock import patch

        from backend.copilot.sdk.service import _read_cli_session_from_disk

        session_id = "12345678-0000-0000-0000-aabbccdd0001"
        projects_base_dir = str(tmp_path)
        sdk_cwd, session_file = self._build_session_file(tmp_path, session_id)

        # Write raw invalid UTF-8 bytes
        session_file.write_bytes(b"\xff\xfe invalid utf-8\n")

        with (
            patch(
                "backend.copilot.sdk.service.projects_base",
                return_value=projects_base_dir,
            ),
            patch(
                "backend.copilot.transcript.projects_base",
                return_value=projects_base_dir,
            ),
        ):
            result = _read_cli_session_from_disk(sdk_cwd, session_id, "[Test]")

        # UnicodeDecodeError path returns the raw bytes (upload-raw fallback)
        assert result == b"\xff\xfe invalid utf-8\n"

    def test_write_back_oserror_still_returns_stripped_bytes(self, tmp_path):
        """OSError on write-back returns stripped bytes for GCS upload (not raw)."""
        from unittest.mock import patch

        from backend.copilot.sdk.service import _read_cli_session_from_disk

        session_id = "12345678-0000-0000-0000-aabbccdd0002"
        projects_base_dir = str(tmp_path)
        sdk_cwd, session_file = self._build_session_file(tmp_path, session_id)

        # Content with a strippable progress entry so stripped_bytes < raw_bytes
        raw_content = (
            '{"type":"progress","uuid":"p1","subtype":"agent_progress","parentUuid":null}\n'
            '{"type":"user","uuid":"u1","parentUuid":null,"message":{"role":"user","content":"hi"}}\n'
            '{"type":"assistant","uuid":"a1","parentUuid":"u1","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}\n'
        )
        session_file.write_bytes(raw_content.encode("utf-8"))
        # Make the file read-only so write_bytes raises OSError on the write-back
        session_file.chmod(0o444)

        try:
            with (
                patch(
                    "backend.copilot.sdk.service.projects_base",
                    return_value=projects_base_dir,
                ),
                patch(
                    "backend.copilot.transcript.projects_base",
                    return_value=projects_base_dir,
                ),
            ):
                result = _read_cli_session_from_disk(sdk_cwd, session_id, "[Test]")
        finally:
            session_file.chmod(0o644)

        # Must return stripped bytes (not raw, not None) so GCS gets the clean version
        assert result is not None
        assert (
            b"progress" not in result
        ), "Stripped bytes must not contain progress entry"
        assert b"hello" in result, "Stripped bytes should contain assistant turn"
