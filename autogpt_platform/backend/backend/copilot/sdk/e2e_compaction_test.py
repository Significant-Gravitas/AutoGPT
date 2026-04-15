"""End-to-end compaction flow test.

Simulates the full service.py compaction lifecycle using real-format
JSONL session files — no SDK subprocess needed. Exercises:

  1. TranscriptBuilder loads a "downloaded" transcript
  2. User query appended, assistant response streamed
  3. PreCompact hook fires → CompactionTracker.on_compact()
  4. Next message → emit_start_if_ready() yields spinner events
  5. Message after that → emit_end_if_ready() returns CompactionResult
  6. read_compacted_entries() reads the CLI session file
  7. TranscriptBuilder.replace_entries() syncs state
  8. More messages appended post-compaction
  9. to_jsonl() exports full state for upload
  10. Fresh builder loads the export — roundtrip verified
"""

import asyncio

from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    StreamFinishStep,
    StreamStartStep,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.sdk.compaction import CompactionTracker
from backend.copilot.sdk.transcript import (
    read_compacted_entries,
    strip_progress_entries,
)
from backend.copilot.sdk.transcript_builder import TranscriptBuilder
from backend.util import json


def _make_jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures: realistic CLI session file content
# ---------------------------------------------------------------------------

# Pre-compaction conversation
USER_1 = {
    "type": "user",
    "uuid": "u1",
    "message": {"role": "user", "content": "What files are in this project?"},
}
ASST_1_THINKING = {
    "type": "assistant",
    "uuid": "a1-think",
    "parentUuid": "u1",
    "message": {
        "role": "assistant",
        "id": "msg_sdk_aaa",
        "type": "message",
        "content": [{"type": "thinking", "thinking": "Let me look at the files..."}],
        "stop_reason": None,
        "stop_sequence": None,
    },
}
ASST_1_TOOL = {
    "type": "assistant",
    "uuid": "a1-tool",
    "parentUuid": "u1",
    "message": {
        "role": "assistant",
        "id": "msg_sdk_aaa",
        "type": "message",
        "content": [
            {
                "type": "tool_use",
                "id": "tu1",
                "name": "Bash",
                "input": {"command": "ls"},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
    },
}
TOOL_RESULT_1 = {
    "type": "user",
    "uuid": "tr1",
    "parentUuid": "a1-tool",
    "message": {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu1",
                "content": "file1.py\nfile2.py",
            }
        ],
    },
}
ASST_1_TEXT = {
    "type": "assistant",
    "uuid": "a1-text",
    "parentUuid": "tr1",
    "message": {
        "role": "assistant",
        "id": "msg_sdk_bbb",
        "type": "message",
        "content": [{"type": "text", "text": "I found file1.py and file2.py."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    },
}
# Progress entries (should be stripped during upload)
PROGRESS_1 = {
    "type": "progress",
    "uuid": "prog1",
    "parentUuid": "a1-tool",
    "data": {"type": "bash_progress", "stdout": "running ls..."},
}
# Second user message
USER_2 = {
    "type": "user",
    "uuid": "u2",
    "parentUuid": "a1-text",
    "message": {"role": "user", "content": "Show me file1.py"},
}
ASST_2 = {
    "type": "assistant",
    "uuid": "a2",
    "parentUuid": "u2",
    "message": {
        "role": "assistant",
        "id": "msg_sdk_ccc",
        "type": "message",
        "content": [{"type": "text", "text": "Here is file1.py content..."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    },
}

# --- Compaction summary (written by CLI after context compaction) ---
COMPACT_SUMMARY = {
    "type": "summary",
    "uuid": "cs1",
    "isCompactSummary": True,
    "message": {
        "role": "user",
        "content": (
            "Summary: User asked about project files. Found file1.py and file2.py. "
            "User then asked to see file1.py."
        ),
    },
}

# Post-compaction assistant response
POST_COMPACT_ASST = {
    "type": "assistant",
    "uuid": "a3",
    "parentUuid": "cs1",
    "message": {
        "role": "assistant",
        "id": "msg_sdk_ddd",
        "type": "message",
        "content": [{"type": "text", "text": "Here is the content of file1.py..."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    },
}

# Post-compaction user follow-up
USER_3 = {
    "type": "user",
    "uuid": "u3",
    "parentUuid": "a3",
    "message": {"role": "user", "content": "Now show file2.py"},
}
ASST_3 = {
    "type": "assistant",
    "uuid": "a4",
    "parentUuid": "u3",
    "message": {
        "role": "assistant",
        "id": "msg_sdk_eee",
        "type": "message",
        "content": [{"type": "text", "text": "Here is file2.py..."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    },
}


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


class TestCompactionE2E:
    def _write_session_file(self, session_dir, entries):
        """Write a CLI session JSONL file."""
        path = session_dir / "session.jsonl"
        path.write_text(_make_jsonl(*entries))
        return path

    def test_full_compaction_lifecycle(self, tmp_path, monkeypatch):
        """Simulate the complete service.py compaction flow.

        Timeline:
        1. Previous turn uploaded transcript with [USER_1, ASST_1, USER_2, ASST_2]
        2. Current turn: download → load_previous
        3. User sends "Now show file2.py" → append_user
        4. SDK starts streaming response
        5. Mid-stream: PreCompact hook fires (context too large)
        6. CLI writes compaction summary to session file
        7. Next SDK message → emit_start (spinner)
        8. Following message → emit_end (CompactionResult)
        9. read_compacted_entries reads the session file
        10. replace_entries syncs TranscriptBuilder
        11. More assistant messages appended
        12. Export → upload → next turn downloads it
        """
        # --- Setup CLI projects directory ---
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        # --- Step 1-2: Load "downloaded" transcript from previous turn ---
        previous_transcript = _make_jsonl(
            USER_1,
            ASST_1_THINKING,
            ASST_1_TOOL,
            TOOL_RESULT_1,
            ASST_1_TEXT,
            USER_2,
            ASST_2,
        )
        builder = TranscriptBuilder()
        builder.load_previous(previous_transcript)
        assert builder.entry_count == 7

        # --- Step 3: User sends new query ---
        builder.append_user("Now show file2.py")
        assert builder.entry_count == 8

        # --- Step 4: SDK starts streaming ---
        builder.append_assistant(
            [{"type": "thinking", "thinking": "Let me read file2.py..."}],
            model="claude-sonnet-4-20250514",
        )
        assert builder.entry_count == 9

        # --- Step 5-6: PreCompact fires, CLI writes session file ---
        session_file = self._write_session_file(
            session_dir,
            [
                USER_1,
                ASST_1_THINKING,
                ASST_1_TOOL,
                PROGRESS_1,
                TOOL_RESULT_1,
                ASST_1_TEXT,
                USER_2,
                ASST_2,
                COMPACT_SUMMARY,
                POST_COMPACT_ASST,
                USER_3,
                ASST_3,
            ],
        )

        # --- Step 7: CompactionTracker receives PreCompact hook ---
        tracker = CompactionTracker()
        session = ChatSession.new(user_id="test-user", dry_run=False)
        tracker.on_compact(str(session_file))

        # --- Step 8: Next SDK message arrives → emit_start ---
        start_events = tracker.emit_start_if_ready()
        assert len(start_events) == 3
        assert isinstance(start_events[0], StreamStartStep)
        assert isinstance(start_events[1], StreamToolInputStart)
        assert isinstance(start_events[2], StreamToolInputAvailable)

        # Verify tool_call_id is set
        tool_call_id = start_events[1].toolCallId
        assert tool_call_id.startswith("compaction-")

        # --- Step 9: Following message → emit_end ---
        result = _run(tracker.emit_end_if_ready(session))
        assert result.just_ended is True
        assert result.transcript_path == str(session_file)
        assert len(result.events) == 2
        assert isinstance(result.events[0], StreamToolOutputAvailable)
        assert isinstance(result.events[1], StreamFinishStep)
        # Verify same tool_call_id
        assert result.events[0].toolCallId == tool_call_id

        # Session should have compaction messages persisted
        assert len(session.messages) == 2
        assert session.messages[0].role == "assistant"
        assert session.messages[1].role == "tool"

        # --- Step 10: read_compacted_entries + replace_entries ---
        compacted = read_compacted_entries(str(session_file))
        assert compacted is not None
        # Should have: COMPACT_SUMMARY + POST_COMPACT_ASST + USER_3 + ASST_3
        assert len(compacted) == 4
        assert compacted[0]["uuid"] == "cs1"
        assert compacted[0]["isCompactSummary"] is True

        # Replace builder state with compacted entries
        old_count = builder.entry_count
        builder.replace_entries(compacted)
        assert builder.entry_count == 4  # Only compacted entries
        assert builder.entry_count < old_count  # Compaction reduced entries

        # --- Step 11: More assistant messages after compaction ---
        builder.append_assistant(
            [{"type": "text", "text": "Here is file2.py:\n\ndef hello():\n    pass"}],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
        )
        assert builder.entry_count == 5

        # --- Step 12: Export for upload ---
        output = builder.to_jsonl()
        assert output  # Not empty
        output_entries = [json.loads(line) for line in output.strip().split("\n")]
        assert len(output_entries) == 5

        # Verify structure:
        # [COMPACT_SUMMARY, POST_COMPACT_ASST, USER_3, ASST_3, new_assistant]
        assert output_entries[0]["type"] == "summary"
        assert output_entries[0].get("isCompactSummary") is True
        assert output_entries[0]["uuid"] == "cs1"
        assert output_entries[1]["uuid"] == "a3"
        assert output_entries[2]["uuid"] == "u3"
        assert output_entries[3]["uuid"] == "a4"
        assert output_entries[4]["type"] == "assistant"

        # Verify parent chain is intact
        assert output_entries[1]["parentUuid"] == "cs1"  # a3 → cs1
        assert output_entries[2]["parentUuid"] == "a3"  # u3 → a3
        assert output_entries[3]["parentUuid"] == "u3"  # a4 → u3
        assert output_entries[4]["parentUuid"] == "a4"  # new → a4

        # --- Step 13: Roundtrip — next turn loads this export ---
        builder2 = TranscriptBuilder()
        builder2.load_previous(output)
        assert builder2.entry_count == 5

        # isCompactSummary survives roundtrip
        output2 = builder2.to_jsonl()
        first_entry = json.loads(output2.strip().split("\n")[0])
        assert first_entry.get("isCompactSummary") is True

        # Can append more messages
        builder2.append_user("What about file3.py?")
        assert builder2.entry_count == 6
        final_output = builder2.to_jsonl()
        last_entry = json.loads(final_output.strip().split("\n")[-1])
        assert last_entry["type"] == "user"
        # Parented to the last entry from previous turn
        assert last_entry["parentUuid"] == output_entries[-1]["uuid"]

    def test_double_compaction_within_session(self, tmp_path, monkeypatch):
        """Two compactions in the same session (across reset_for_query)."""
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        tracker = CompactionTracker()
        session = ChatSession.new(user_id="test", dry_run=False)
        builder = TranscriptBuilder()

        # --- First query with compaction ---
        builder.append_user("first question")
        builder.append_assistant([{"type": "text", "text": "first answer"}])

        # Write session file for first compaction
        first_summary = {
            "type": "summary",
            "uuid": "cs-first",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "First compaction summary"},
        }
        first_post = {
            "type": "assistant",
            "uuid": "a-first",
            "parentUuid": "cs-first",
            "message": {"role": "assistant", "content": "first post-compact"},
        }
        file1 = session_dir / "session1.jsonl"
        file1.write_text(_make_jsonl(first_summary, first_post))

        tracker.on_compact(str(file1))
        tracker.emit_start_if_ready()
        result1 = _run(tracker.emit_end_if_ready(session))
        assert result1.just_ended is True

        compacted1 = read_compacted_entries(str(file1))
        assert compacted1 is not None
        builder.replace_entries(compacted1)
        assert builder.entry_count == 2

        # --- Reset for second query ---
        tracker.reset_for_query()

        # --- Second query with compaction ---
        builder.append_user("second question")
        builder.append_assistant([{"type": "text", "text": "second answer"}])

        second_summary = {
            "type": "summary",
            "uuid": "cs-second",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "Second compaction summary"},
        }
        second_post = {
            "type": "assistant",
            "uuid": "a-second",
            "parentUuid": "cs-second",
            "message": {"role": "assistant", "content": "second post-compact"},
        }
        file2 = session_dir / "session2.jsonl"
        file2.write_text(_make_jsonl(second_summary, second_post))

        tracker.on_compact(str(file2))
        tracker.emit_start_if_ready()
        result2 = _run(tracker.emit_end_if_ready(session))
        assert result2.just_ended is True

        compacted2 = read_compacted_entries(str(file2))
        assert compacted2 is not None
        builder.replace_entries(compacted2)
        assert builder.entry_count == 2  # Only second compaction entries

        # Export and verify
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[0]["uuid"] == "cs-second"
        assert entries[0].get("isCompactSummary") is True

    def test_strip_progress_then_load_then_compact_roundtrip(
        self, tmp_path, monkeypatch
    ):
        """Full pipeline: strip → load → compact → replace → export → reload.

        This tests the exact sequence that happens across two turns:
        Turn 1: SDK produces transcript with progress entries
        Upload: strip_progress_entries removes progress, upload to cloud
        Turn 2: Download → load_previous → compaction fires → replace → export
        Turn 3: Download the Turn 2 export → load_previous (roundtrip)
        """
        config_dir = tmp_path / "config"
        projects_dir = config_dir / "projects"
        session_dir = projects_dir / "proj"
        session_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

        # --- Turn 1: SDK produces raw transcript ---
        raw_content = _make_jsonl(
            USER_1,
            ASST_1_THINKING,
            ASST_1_TOOL,
            PROGRESS_1,
            TOOL_RESULT_1,
            ASST_1_TEXT,
            USER_2,
            ASST_2,
        )

        # Strip progress for upload
        stripped = strip_progress_entries(raw_content)
        stripped_entries = [
            json.loads(line) for line in stripped.strip().split("\n") if line.strip()
        ]
        # Progress should be gone
        assert not any(e.get("type") == "progress" for e in stripped_entries)
        assert len(stripped_entries) == 7  # 8 - 1 progress

        # --- Turn 2: Download stripped, load, compaction happens ---
        builder = TranscriptBuilder()
        builder.load_previous(stripped)
        assert builder.entry_count == 7

        builder.append_user("Now show file2.py")
        builder.append_assistant(
            [{"type": "text", "text": "Reading file2.py..."}],
            model="claude-sonnet-4-20250514",
        )

        # CLI writes session file with compaction
        session_file = self._write_session_file(
            session_dir,
            [
                USER_1,
                ASST_1_TOOL,
                TOOL_RESULT_1,
                ASST_1_TEXT,
                USER_2,
                ASST_2,
                COMPACT_SUMMARY,
                POST_COMPACT_ASST,
            ],
        )

        compacted = read_compacted_entries(str(session_file))
        assert compacted is not None
        builder.replace_entries(compacted)

        # Append post-compaction message
        builder.append_user("Thanks!")
        output = builder.to_jsonl()

        # --- Turn 3: Fresh load of Turn 2 export ---
        builder3 = TranscriptBuilder()
        builder3.load_previous(output)
        # Should have: compact_summary + post_compact_asst + "Thanks!"
        assert builder3.entry_count == 3

        # Compact summary survived the full pipeline
        first = json.loads(builder3.to_jsonl().strip().split("\n")[0])
        assert first.get("isCompactSummary") is True
        assert first["type"] == "summary"
