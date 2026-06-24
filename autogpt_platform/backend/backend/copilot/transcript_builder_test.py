"""Tests for canonical TranscriptBuilder (backend.copilot.transcript_builder).

These tests directly import from the canonical module to ensure codecov
patch coverage for the new file.
"""

from backend.copilot.transcript_builder import TranscriptBuilder, TranscriptEntry
from backend.util import json


def _make_jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


USER_MSG = {
    "type": "user",
    "uuid": "u1",
    "message": {"role": "user", "content": "hello"},
}
ASST_MSG = {
    "type": "assistant",
    "uuid": "a1",
    "parentUuid": "u1",
    "message": {
        "role": "assistant",
        "id": "msg_1",
        "type": "message",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    },
}


class TestTranscriptEntry:
    def test_basic_construction(self):
        entry = TranscriptEntry(
            type="user", uuid="u1", message={"role": "user", "content": "hi"}
        )
        assert entry.type == "user"
        assert entry.uuid == "u1"
        assert entry.parentUuid == ""
        assert entry.isCompactSummary is None

    def test_optional_fields(self):
        entry = TranscriptEntry(
            type="summary",
            uuid="s1",
            parentUuid="p1",
            isCompactSummary=True,
            message={"role": "user", "content": "summary"},
        )
        assert entry.isCompactSummary is True
        assert entry.parentUuid == "p1"


class TestTranscriptBuilderInit:
    def test_starts_empty(self):
        builder = TranscriptBuilder()
        assert builder.is_empty
        assert builder.entry_count == 0
        assert builder.last_entry_type is None
        assert builder.to_jsonl() == ""


class TestAppendUser:
    def test_appends_user_entry(self):
        builder = TranscriptBuilder()
        builder.append_user("hello")
        assert builder.entry_count == 1
        assert builder.last_entry_type == "user"

    def test_chains_parent_uuid(self):
        builder = TranscriptBuilder()
        builder.append_user("first", uuid="u1")
        builder.append_user("second", uuid="u2")
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[0]["parentUuid"] == ""
        assert entries[1]["parentUuid"] == "u1"

    def test_custom_uuid(self):
        builder = TranscriptBuilder()
        builder.append_user("hello", uuid="custom-id")
        output = builder.to_jsonl()
        entry = json.loads(output.strip())
        assert entry["uuid"] == "custom-id"


class TestAppendToolResult:
    def test_appends_as_user_entry(self):
        builder = TranscriptBuilder()
        builder.append_tool_result(tool_use_id="tc_1", content="result text")
        assert builder.entry_count == 1
        assert builder.last_entry_type == "user"
        output = builder.to_jsonl()
        entry = json.loads(output.strip())
        content = entry["message"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tc_1"
        assert content[0]["content"] == "result text"


class TestAppendAssistant:
    def test_appends_assistant_entry(self):
        builder = TranscriptBuilder()
        builder.append_user("hi")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "hello"}],
            model="test-model",
            stop_reason="end_turn",
        )
        assert builder.entry_count == 2
        assert builder.last_entry_type == "assistant"

    def test_consecutive_assistants_share_message_id(self):
        builder = TranscriptBuilder()
        builder.append_user("hi")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "part 1"}],
            model="m",
        )
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "part 2"}],
            model="m",
        )
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        # The two assistant entries share the same message ID
        assert entries[1]["message"]["id"] == entries[2]["message"]["id"]

    def test_non_consecutive_assistants_get_different_ids(self):
        builder = TranscriptBuilder()
        builder.append_user("q1")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "a1"}],
            model="m",
        )
        builder.append_user("q2")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "a2"}],
            model="m",
        )
        output = builder.to_jsonl()
        entries = [json.loads(line) for line in output.strip().split("\n")]
        assert entries[1]["message"]["id"] != entries[3]["message"]["id"]


class TestLoadPrevious:
    def test_loads_valid_entries(self):
        content = _make_jsonl(USER_MSG, ASST_MSG)
        builder = TranscriptBuilder()
        builder.load_previous(content)
        assert builder.entry_count == 2

    def test_skips_empty_content(self):
        builder = TranscriptBuilder()
        builder.load_previous("")
        assert builder.is_empty
        builder.load_previous("   ")
        assert builder.is_empty

    def test_skips_strippable_types(self):
        progress = {"type": "progress", "uuid": "p1", "message": {}}
        content = _make_jsonl(USER_MSG, progress, ASST_MSG)
        builder = TranscriptBuilder()
        builder.load_previous(content)
        assert builder.entry_count == 2  # progress was skipped

    def test_preserves_compact_summary(self):
        compact = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "summary"},
        }
        content = _make_jsonl(compact, ASST_MSG)
        builder = TranscriptBuilder()
        builder.load_previous(content)
        assert builder.entry_count == 2

    def test_skips_invalid_json_lines(self):
        content = '{"type":"user","uuid":"u1","message":{}}\nnot-valid-json\n'
        builder = TranscriptBuilder()
        builder.load_previous(content)
        assert builder.entry_count == 1


class TestToJsonl:
    def test_roundtrip(self):
        builder = TranscriptBuilder()
        builder.append_user("hello", uuid="u1")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "world"}],
            model="m",
        )
        output = builder.to_jsonl()
        assert output.endswith("\n")
        lines = output.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "type" in parsed
            assert "uuid" in parsed
            assert "message" in parsed


class TestReplaceEntries:
    def test_replaces_all_entries(self):
        builder = TranscriptBuilder()
        builder.append_user("old")
        builder.append_assistant(
            content_blocks=[{"type": "text", "text": "old answer"}], model="m"
        )
        assert builder.entry_count == 2

        compacted = [
            {
                "type": "summary",
                "uuid": "cs1",
                "isCompactSummary": True,
                "message": {"role": "user", "content": "compacted"},
            }
        ]
        builder.replace_entries(compacted)
        assert builder.entry_count == 1

    def test_empty_replacement_keeps_existing(self):
        builder = TranscriptBuilder()
        builder.append_user("keep me")
        builder.replace_entries([])
        assert builder.entry_count == 1


class TestParseEntry:
    def test_filters_strippable_non_compact(self):
        result = TranscriptBuilder._parse_entry(
            {"type": "progress", "uuid": "p1", "message": {}}
        )
        assert result is None

    def test_keeps_compact_summary(self):
        result = TranscriptBuilder._parse_entry(
            {
                "type": "summary",
                "uuid": "cs1",
                "isCompactSummary": True,
                "message": {},
            }
        )
        assert result is not None
        assert result.isCompactSummary is True

    def test_generates_uuid_if_missing(self):
        result = TranscriptBuilder._parse_entry(
            {"type": "user", "message": {"role": "user", "content": "hi"}}
        )
        assert result is not None
        assert result.uuid  # Should be a generated UUID
