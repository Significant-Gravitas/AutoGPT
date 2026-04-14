"""Tests for canonical transcript module (backend.copilot.transcript).

Covers pure helper functions that are not exercised by the SDK re-export tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.util import json

from .transcript import (
    TranscriptDownload,
    _build_path_from_parts,
    _find_last_assistant_entry,
    _flatten_assistant_content,
    _flatten_tool_result_content,
    _messages_to_transcript,
    _meta_storage_path_parts,
    _rechain_tail,
    _sanitize_id,
    _storage_path_parts,
    _transcript_to_messages,
    strip_for_upload,
    validate_transcript,
)


def _make_jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


# ---------------------------------------------------------------------------
# _sanitize_id
# ---------------------------------------------------------------------------


class TestSanitizeId:
    def test_uuid_passes_through(self):
        assert _sanitize_id("abcdef12-3456-7890-abcd-ef1234567890") == (
            "abcdef12-3456-7890-abcd-ef1234567890"
        )

    def test_strips_non_hex_characters(self):
        # Only hex chars (0-9, a-f, A-F) and hyphens are kept
        result = _sanitize_id("abc/../../etc/passwd")
        assert "/" not in result
        assert "." not in result
        # 'p', 's', 'w' are not hex chars, so they are stripped
        assert all(c in "0123456789abcdefABCDEF-" for c in result)

    def test_truncates_to_max_len(self):
        long_id = "a" * 100
        result = _sanitize_id(long_id, max_len=10)
        assert len(result) == 10

    def test_empty_returns_unknown(self):
        assert _sanitize_id("") == "unknown"

    def test_none_returns_unknown(self):
        assert _sanitize_id(None) == "unknown"  # type: ignore[arg-type]

    def test_special_chars_only_returns_unknown(self):
        assert _sanitize_id("!@#$%^&*()") == "unknown"


# ---------------------------------------------------------------------------
# _storage_path_parts / _meta_storage_path_parts
# ---------------------------------------------------------------------------


class TestStoragePathParts:
    def test_returns_triple(self):
        prefix, uid, fname = _storage_path_parts("user-1", "sess-2")
        assert prefix == "chat-transcripts"
        assert "e" in uid  # hex chars from "user-1" sanitized
        assert fname.endswith(".jsonl")

    def test_meta_returns_meta_json(self):
        prefix, _, fname = _meta_storage_path_parts("user-1", "sess-2")
        assert prefix == "chat-transcripts"
        assert fname.endswith(".meta.json")


# ---------------------------------------------------------------------------
# _build_path_from_parts
# ---------------------------------------------------------------------------


class TestBuildPathFromParts:
    def test_gcs_backend(self):
        from backend.util.workspace_storage import GCSWorkspaceStorage

        mock_gcs = MagicMock(spec=GCSWorkspaceStorage)
        mock_gcs.bucket_name = "my-bucket"
        path = _build_path_from_parts(("wid", "fid", "file.jsonl"), mock_gcs)
        assert path == "gcs://my-bucket/workspaces/wid/fid/file.jsonl"

    def test_local_backend(self):
        # Use a plain object (not MagicMock) so isinstance(GCSWorkspaceStorage) is False
        local_backend = type("LocalBackend", (), {})()
        path = _build_path_from_parts(("wid", "fid", "file.jsonl"), local_backend)
        assert path == "local://wid/fid/file.jsonl"


# ---------------------------------------------------------------------------
# TranscriptDownload dataclass
# ---------------------------------------------------------------------------


class TestTranscriptDownload:
    def test_defaults(self):
        td = TranscriptDownload(content="hello")
        assert td.content == "hello"
        assert td.message_count == 0
        assert td.uploaded_at == 0.0

    def test_custom_values(self):
        td = TranscriptDownload(content="data", message_count=5, uploaded_at=123.45)
        assert td.message_count == 5
        assert td.uploaded_at == 123.45


# ---------------------------------------------------------------------------
# _flatten_assistant_content
# ---------------------------------------------------------------------------


class TestFlattenAssistantContent:
    def test_text_blocks(self):
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert _flatten_assistant_content(blocks) == "Hello\nWorld"

    def test_thinking_blocks_stripped(self):
        blocks = [
            {"type": "thinking", "thinking": "hmm..."},
            {"type": "text", "text": "answer"},
            {"type": "redacted_thinking", "data": "secret"},
        ]
        assert _flatten_assistant_content(blocks) == "answer"

    def test_tool_use_blocks_stripped(self):
        blocks = [
            {"type": "text", "text": "I'll run a tool"},
            {"type": "tool_use", "name": "bash", "id": "tc1", "input": {}},
        ]
        assert _flatten_assistant_content(blocks) == "I'll run a tool"

    def test_string_blocks(self):
        blocks = ["hello", "world"]
        assert _flatten_assistant_content(blocks) == "hello\nworld"

    def test_empty_blocks(self):
        assert _flatten_assistant_content([]) == ""

    def test_unknown_dict_blocks_skipped(self):
        blocks = [{"type": "image", "data": "base64..."}]
        assert _flatten_assistant_content(blocks) == ""


# ---------------------------------------------------------------------------
# _flatten_tool_result_content
# ---------------------------------------------------------------------------


class TestFlattenToolResultContent:
    def test_tool_result_with_text_content(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "tc1",
                "content": [{"type": "text", "text": "output data"}],
            }
        ]
        assert _flatten_tool_result_content(blocks) == "output data"

    def test_tool_result_with_string_content(self):
        blocks = [
            {"type": "tool_result", "tool_use_id": "tc1", "content": "simple string"}
        ]
        assert _flatten_tool_result_content(blocks) == "simple string"

    def test_tool_result_with_image_placeholder(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "tc1",
                "content": [{"type": "image", "data": "base64..."}],
            }
        ]
        assert _flatten_tool_result_content(blocks) == "[__image__]"

    def test_tool_result_with_document_placeholder(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "tc1",
                "content": [{"type": "document", "data": "base64..."}],
            }
        ]
        assert _flatten_tool_result_content(blocks) == "[__document__]"

    def test_tool_result_with_none_content(self):
        blocks = [{"type": "tool_result", "tool_use_id": "tc1", "content": None}]
        assert _flatten_tool_result_content(blocks) == ""

    def test_text_block_outside_tool_result(self):
        blocks = [{"type": "text", "text": "standalone"}]
        assert _flatten_tool_result_content(blocks) == "standalone"

    def test_unknown_dict_block_placeholder(self):
        blocks = [{"type": "custom_widget", "data": "x"}]
        assert _flatten_tool_result_content(blocks) == "[__custom_widget__]"

    def test_string_blocks(self):
        blocks = ["raw text"]
        assert _flatten_tool_result_content(blocks) == "raw text"

    def test_empty_blocks(self):
        assert _flatten_tool_result_content([]) == ""

    def test_mixed_content_in_tool_result(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "tc1",
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "image", "data": "..."},
                    "raw string",
                ],
            }
        ]
        result = _flatten_tool_result_content(blocks)
        assert "line1" in result
        assert "[__image__]" in result
        assert "raw string" in result

    def test_tool_result_with_dict_without_text_key(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "tc1",
                "content": [{"count": 42}],
            }
        ]
        result = _flatten_tool_result_content(blocks)
        assert "42" in result

    def test_tool_result_content_list_with_list_content(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "tc1",
                "content": [{"type": "text", "text": None}],
            }
        ]
        result = _flatten_tool_result_content(blocks)
        assert result == "None"


# ---------------------------------------------------------------------------
# _transcript_to_messages
# ---------------------------------------------------------------------------

USER_ENTRY = {
    "type": "user",
    "uuid": "u1",
    "parentUuid": "",
    "message": {"role": "user", "content": "hello"},
}
ASST_ENTRY = {
    "type": "assistant",
    "uuid": "a1",
    "parentUuid": "u1",
    "message": {
        "role": "assistant",
        "id": "msg_1",
        "content": [{"type": "text", "text": "hi there"}],
    },
}
PROGRESS_ENTRY = {
    "type": "progress",
    "uuid": "p1",
    "parentUuid": "u1",
    "data": {},
}


class TestTranscriptToMessages:
    def test_basic_conversion(self):
        content = _make_jsonl(USER_ENTRY, ASST_ENTRY)
        messages = _transcript_to_messages(content)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "hi there"

    def test_skips_strippable_types(self):
        content = _make_jsonl(USER_ENTRY, PROGRESS_ENTRY, ASST_ENTRY)
        messages = _transcript_to_messages(content)
        assert len(messages) == 2

    def test_skips_entries_without_role(self):
        no_role = {"type": "user", "uuid": "x", "message": {"content": "no role"}}
        content = _make_jsonl(no_role)
        messages = _transcript_to_messages(content)
        assert len(messages) == 0

    def test_handles_string_content(self):
        entry = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "plain string"},
        }
        content = _make_jsonl(entry)
        messages = _transcript_to_messages(content)
        assert messages[0]["content"] == "plain string"

    def test_handles_tool_result_content(self):
        entry = {
            "type": "user",
            "uuid": "u1",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tc1", "content": "output"}
                ],
            },
        }
        content = _make_jsonl(entry)
        messages = _transcript_to_messages(content)
        assert messages[0]["content"] == "output"

    def test_handles_none_content(self):
        entry = {
            "type": "assistant",
            "uuid": "a1",
            "message": {"role": "assistant", "content": None},
        }
        content = _make_jsonl(entry)
        messages = _transcript_to_messages(content)
        assert messages[0]["content"] == ""

    def test_skips_invalid_json(self):
        content = "not valid json\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 0

    def test_preserves_compact_summary(self):
        compact = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "summary of conversation"},
        }
        content = _make_jsonl(compact)
        messages = _transcript_to_messages(content)
        assert len(messages) == 1

    def test_strips_summary_without_compact_flag(self):
        summary = {
            "type": "summary",
            "uuid": "s1",
            "message": {"role": "user", "content": "summary"},
        }
        content = _make_jsonl(summary)
        messages = _transcript_to_messages(content)
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# _messages_to_transcript
# ---------------------------------------------------------------------------


class TestMessagesToTranscript:
    def test_basic_roundtrip(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = _messages_to_transcript(messages)
        assert result.endswith("\n")
        lines = result.strip().split("\n")
        assert len(lines) == 2

        user_entry = json.loads(lines[0])
        assert user_entry["type"] == "user"
        assert user_entry["message"]["role"] == "user"
        assert user_entry["message"]["content"] == "hello"
        assert user_entry["parentUuid"] == ""

        asst_entry = json.loads(lines[1])
        assert asst_entry["type"] == "assistant"
        assert asst_entry["message"]["role"] == "assistant"
        assert asst_entry["message"]["content"] == [{"type": "text", "text": "world"}]
        assert asst_entry["parentUuid"] == user_entry["uuid"]

    def test_empty_messages(self):
        assert _messages_to_transcript([]) == ""

    def test_assistant_has_message_envelope(self):
        messages = [{"role": "assistant", "content": "test"}]
        result = _messages_to_transcript(messages)
        entry = json.loads(result.strip())
        msg = entry["message"]
        assert "id" in msg
        assert msg["id"].startswith("msg_compact_")
        assert msg["type"] == "message"
        assert msg["stop_reason"] == "end_turn"
        assert msg["stop_sequence"] is None

    def test_uuid_chain(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = _messages_to_transcript(messages)
        lines = result.strip().split("\n")
        entries = [json.loads(line) for line in lines]
        assert entries[0]["parentUuid"] == ""
        assert entries[1]["parentUuid"] == entries[0]["uuid"]
        assert entries[2]["parentUuid"] == entries[1]["uuid"]

    def test_assistant_with_empty_content(self):
        messages = [{"role": "assistant", "content": ""}]
        result = _messages_to_transcript(messages)
        entry = json.loads(result.strip())
        assert entry["message"]["content"] == []


# ---------------------------------------------------------------------------
# _find_last_assistant_entry
# ---------------------------------------------------------------------------


class TestFindLastAssistantEntry:
    def test_splits_at_last_assistant(self):
        user = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "hi"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "message": {"role": "assistant", "id": "msg1", "content": "answer"},
        }
        content = _make_jsonl(user, asst)
        prefix, tail = _find_last_assistant_entry(content)
        assert len(prefix) == 1
        assert len(tail) == 1

    def test_no_assistant_returns_all_in_prefix(self):
        user1 = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "hi"},
        }
        user2 = {
            "type": "user",
            "uuid": "u2",
            "message": {"role": "user", "content": "hey"},
        }
        content = _make_jsonl(user1, user2)
        prefix, tail = _find_last_assistant_entry(content)
        assert len(prefix) == 2
        assert len(tail) == 0

    def test_multi_entry_turn_preserved(self):
        user = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "q"},
        }
        asst1 = {
            "type": "assistant",
            "uuid": "a1",
            "message": {
                "role": "assistant",
                "id": "msg_turn",
                "content": [{"type": "thinking", "thinking": "hmm"}],
            },
        }
        asst2 = {
            "type": "assistant",
            "uuid": "a2",
            "message": {
                "role": "assistant",
                "id": "msg_turn",
                "content": [{"type": "text", "text": "answer"}],
            },
        }
        content = _make_jsonl(user, asst1, asst2)
        prefix, tail = _find_last_assistant_entry(content)
        assert len(prefix) == 1  # just the user
        assert len(tail) == 2  # both assistant entries

    def test_assistant_without_id(self):
        user = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "q"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "message": {"role": "assistant", "content": "no id"},
        }
        content = _make_jsonl(user, asst)
        prefix, tail = _find_last_assistant_entry(content)
        assert len(prefix) == 1
        assert len(tail) == 1

    def test_trailing_user_after_assistant(self):
        user1 = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "q"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "message": {"role": "assistant", "id": "msg1", "content": "a"},
        }
        user2 = {
            "type": "user",
            "uuid": "u2",
            "message": {"role": "user", "content": "follow"},
        }
        content = _make_jsonl(user1, asst, user2)
        prefix, tail = _find_last_assistant_entry(content)
        assert len(prefix) == 1  # user1
        assert len(tail) == 2  # asst + user2


# ---------------------------------------------------------------------------
# _rechain_tail
# ---------------------------------------------------------------------------


class TestRechainTail:
    def test_empty_tail(self):
        assert _rechain_tail("some prefix\n", []) == ""

    def test_patches_first_entry_parent(self):
        prefix_entry = {"uuid": "last-prefix-uuid", "type": "user", "message": {}}
        prefix = json.dumps(prefix_entry) + "\n"

        tail_entry = {
            "uuid": "t1",
            "parentUuid": "old-parent",
            "type": "assistant",
            "message": {},
        }
        tail_lines = [json.dumps(tail_entry)]

        result = _rechain_tail(prefix, tail_lines)
        parsed = json.loads(result.strip())
        assert parsed["parentUuid"] == "last-prefix-uuid"

    def test_chains_consecutive_tail_entries(self):
        prefix_entry = {"uuid": "p1", "type": "user", "message": {}}
        prefix = json.dumps(prefix_entry) + "\n"

        t1 = {"uuid": "t1", "parentUuid": "old1", "type": "assistant", "message": {}}
        t2 = {"uuid": "t2", "parentUuid": "old2", "type": "user", "message": {}}
        tail_lines = [json.dumps(t1), json.dumps(t2)]

        result = _rechain_tail(prefix, tail_lines)
        entries = [json.loads(line) for line in result.strip().split("\n")]
        assert entries[0]["parentUuid"] == "p1"
        assert entries[1]["parentUuid"] == "t1"

    def test_non_dict_lines_passed_through(self):
        prefix_entry = {"uuid": "p1", "type": "user", "message": {}}
        prefix = json.dumps(prefix_entry) + "\n"

        tail_lines = ["not-a-json-dict"]
        result = _rechain_tail(prefix, tail_lines)
        assert "not-a-json-dict" in result


# ---------------------------------------------------------------------------
# strip_for_upload (combined single-parse)
# ---------------------------------------------------------------------------


class TestStripForUpload:
    def test_strips_progress_and_thinking(self):
        user = {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "",
            "message": {"role": "user", "content": "hi"},
        }
        progress = {"type": "progress", "uuid": "p1", "parentUuid": "u1", "data": {}}
        asst_old = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "p1",
            "message": {
                "role": "assistant",
                "id": "msg_old",
                "content": [
                    {"type": "thinking", "thinking": "stale thinking"},
                    {"type": "text", "text": "old answer"},
                ],
            },
        }
        user2 = {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "message": {"role": "user", "content": "next"},
        }
        asst_new = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "message": {
                "role": "assistant",
                "id": "msg_new",
                "content": [
                    {"type": "thinking", "thinking": "fresh thinking"},
                    {"type": "text", "text": "new answer"},
                ],
            },
        }
        content = _make_jsonl(user, progress, asst_old, user2, asst_new)
        result = strip_for_upload(content)

        lines = result.strip().split("\n")
        # Progress should be stripped -> 4 entries remain
        assert len(lines) == 4

        # First entry (user) should be reparented since its child (progress) was stripped
        entries = [json.loads(line) for line in lines]
        types = [e.get("type") for e in entries]
        assert "progress" not in types

        # Old assistant thinking stripped, new assistant thinking preserved
        old_asst = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_old"
        )
        old_content = old_asst["message"]["content"]
        old_types = [b["type"] for b in old_content if isinstance(b, dict)]
        assert "thinking" not in old_types
        assert "text" in old_types

        new_asst = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_new"
        )
        new_content = new_asst["message"]["content"]
        new_types = [b["type"] for b in new_content if isinstance(b, dict)]
        assert "thinking" in new_types  # last assistant preserved

    def test_empty_content(self):
        result = strip_for_upload("")
        # Empty string produces a single empty line after split, resulting in "\n"
        assert result.strip() == ""

    def test_preserves_compact_summary(self):
        compact = {
            "type": "summary",
            "uuid": "cs1",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "summary"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "cs1",
            "message": {"role": "assistant", "id": "msg1", "content": "answer"},
        }
        content = _make_jsonl(compact, asst)
        result = strip_for_upload(content)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_no_assistant_entries(self):
        user = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "hi"},
        }
        content = _make_jsonl(user)
        result = strip_for_upload(content)
        lines = result.strip().split("\n")
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# validate_transcript (additional edge cases)
# ---------------------------------------------------------------------------


class TestValidateTranscript:
    def test_valid_with_assistant(self):
        content = _make_jsonl(
            USER_ENTRY,
            ASST_ENTRY,
        )
        assert validate_transcript(content) is True

    def test_none_returns_false(self):
        assert validate_transcript(None) is False

    def test_whitespace_only_returns_false(self):
        assert validate_transcript("   \n  ") is False

    def test_no_assistant_returns_false(self):
        content = _make_jsonl(USER_ENTRY)
        assert validate_transcript(content) is False

    def test_invalid_json_returns_false(self):
        assert validate_transcript("not json\n") is False

    def test_assistant_only_is_valid(self):
        content = _make_jsonl(ASST_ENTRY)
        assert validate_transcript(content) is True


# ---------------------------------------------------------------------------
# CLI native session file helpers
# ---------------------------------------------------------------------------


class TestCliSessionPath:
    def test_encodes_slashes_to_dashes(self):
        from .transcript import _cli_session_path, _projects_base

        sdk_cwd = "/tmp/copilot-abc"
        result = _cli_session_path(sdk_cwd, "12345678-1234-1234-1234-123456789abc")
        base = _projects_base()
        assert result.startswith(base)
        # Encoded cwd replaces '/' with '-'
        assert "-tmp-copilot-abc" in result
        assert result.endswith(".jsonl")

    def test_sanitizes_session_id(self):
        from .transcript import _cli_session_path

        result = _cli_session_path("/tmp/cwd", "../../etc/passwd")
        # _sanitize_id strips non-hex/hyphen chars; path traversal impossible
        assert ".." not in result
        assert "passwd" not in result


class TestUploadCliSession:
    def test_skips_upload_when_path_outside_projects_base(self, tmp_path):
        """Files outside the CLI projects base are rejected without upload."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_cli_session

        mock_storage = AsyncMock()

        with (
            patch(
                "backend.copilot.transcript._projects_base",
                return_value=str(tmp_path),
            ),
            # Return a path that is genuinely outside tmp_path so that
            # realpath(session_file).startswith(projects_base + "/") is False
            # and the boundary guard actually fires.
            patch(
                "backend.copilot.transcript._cli_session_path",
                return_value="/outside/escaped/session.jsonl",
            ),
            patch(
                "backend.copilot.transcript.get_workspace_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
        ):
            asyncio.run(
                upload_cli_session(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000000",
                    sdk_cwd=str(tmp_path),
                )
            )

        # storage.store must NOT be called — boundary guard should reject the path
        mock_storage.store.assert_not_called()

    def test_skips_upload_when_file_not_found(self, tmp_path):
        """Missing CLI session file logs debug and skips upload silently."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_cli_session

        mock_storage = AsyncMock()
        projects_base = str(tmp_path)

        with (
            patch(
                "backend.copilot.transcript._projects_base",
                return_value=projects_base,
            ),
            patch(
                "backend.copilot.transcript.get_workspace_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
        ):
            # session file doesn't exist — should not raise
            asyncio.run(
                upload_cli_session(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000000",
                    sdk_cwd=str(tmp_path),
                )
            )

        mock_storage.store.assert_not_called()

    def test_uploads_file_successfully(self, tmp_path):
        """Happy path: session file exists within projects base → upload called."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import _sanitize_id, upload_cli_session

        projects_base = str(tmp_path)
        session_id = "12345678-0000-0000-0000-000000000001"
        sdk_cwd = str(tmp_path)

        # Build the path the same way _cli_session_path does, but using our tmp_path
        # as projects_base so the boundary check passes.
        # Must use the same encoding: re.sub non-alphanumeric → "-" on realpath.
        import os
        import re

        encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(sdk_cwd))
        session_dir = tmp_path / encoded_cwd
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / f"{_sanitize_id(session_id)}.jsonl"
        session_file.write_bytes(b'{"type":"assistant"}\n')

        mock_storage = AsyncMock()

        with (
            patch(
                "backend.copilot.transcript._projects_base",
                return_value=projects_base,
            ),
            patch(
                "backend.copilot.transcript.get_workspace_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
        ):
            asyncio.run(
                upload_cli_session(
                    user_id="user-1",
                    session_id=session_id,
                    sdk_cwd=sdk_cwd,
                )
            )

        mock_storage.store.assert_called_once()

    def test_skips_upload_on_oserror(self, tmp_path):
        """OSError reading session file is logged as warning; upload is skipped."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import _sanitize_id, upload_cli_session

        projects_base = str(tmp_path)
        sdk_cwd = str(tmp_path)
        session_id = "12345678-0000-0000-0000-000000000002"

        # Build file at a path inside projects_base so boundary check passes.
        import os
        import re

        encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(sdk_cwd))
        session_dir = tmp_path / encoded_cwd
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / f"{_sanitize_id(session_id)}.jsonl"
        session_file.write_bytes(b'{"type":"assistant"}\n')
        # Remove read permission to trigger OSError
        session_file.chmod(0o000)

        mock_storage = AsyncMock()

        try:
            with (
                patch(
                    "backend.copilot.transcript._projects_base",
                    return_value=projects_base,
                ),
                patch(
                    "backend.copilot.transcript.get_workspace_storage",
                    new_callable=AsyncMock,
                    return_value=mock_storage,
                ),
            ):
                asyncio.run(
                    upload_cli_session(
                        user_id="user-1",
                        session_id=session_id,
                        sdk_cwd=sdk_cwd,
                    )
                )
        finally:
            session_file.chmod(0o644)  # restore so tmp_path cleanup works

        mock_storage.store.assert_not_called()


class TestRestoreCliSession:
    def test_returns_false_when_file_not_found_in_storage(self):
        """Returns False (graceful degradation) when the session is missing."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import restore_cli_session

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = FileNotFoundError("not found")

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                restore_cli_session(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000000",
                    sdk_cwd="/tmp/copilot-test",
                )
            )

        assert result is False

    def test_returns_false_when_restore_path_outside_projects_base(self, tmp_path):
        """Path traversal guard: rejects restoration outside the projects base."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import restore_cli_session

        mock_storage = AsyncMock()
        mock_storage.retrieve.return_value = b'{"type":"assistant"}\n'

        with (
            patch(
                "backend.copilot.transcript.get_workspace_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
            patch(
                "backend.copilot.transcript._projects_base",
                return_value=str(tmp_path),
            ),
            # Return a path genuinely outside tmp_path so the boundary guard fires.
            patch(
                "backend.copilot.transcript._cli_session_path",
                return_value="/outside/escaped/session.jsonl",
            ),
        ):
            result = asyncio.run(
                restore_cli_session(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000000",
                    sdk_cwd=str(tmp_path),
                )
            )

        assert result is False

    def test_returns_true_when_local_file_already_exists(self, tmp_path):
        """Same-pod reuse: if local file exists, skip storage download and return True."""
        import asyncio
        import os
        import re
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from .transcript import restore_cli_session

        session_id = "12345678-0000-0000-0000-000000000099"
        sdk_cwd = str(tmp_path)

        # Pre-create the local session file (simulates previous turn on same pod)
        projects_base = os.path.realpath(str(tmp_path))
        encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", projects_base)
        session_dir = Path(projects_base) / encoded_cwd
        session_dir.mkdir(parents=True, exist_ok=True)
        existing_content = b'{"type":"user"}\n{"type":"assistant"}\n'
        (session_dir / f"{session_id}.jsonl").write_bytes(existing_content)

        mock_storage = AsyncMock()

        with (
            patch(
                "backend.copilot.transcript.get_workspace_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
            patch(
                "backend.copilot.transcript._projects_base",
                return_value=projects_base,
            ),
        ):
            result = asyncio.run(
                restore_cli_session(
                    user_id="user-1",
                    session_id=session_id,
                    sdk_cwd=sdk_cwd,
                )
            )

        assert result is True
        # Storage should NOT have been accessed (local file was used as-is)
        mock_storage.retrieve.assert_not_called()
        # Local file should be unchanged
        assert (session_dir / f"{session_id}.jsonl").read_bytes() == existing_content

    def test_returns_true_on_success(self, tmp_path):
        """Happy path: storage has the session → file written → returns True."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import restore_cli_session

        projects_base = str(tmp_path)
        sdk_cwd = str(tmp_path)
        session_id = "12345678-0000-0000-0000-000000000003"
        content = b'{"type":"assistant"}\n'

        mock_storage = AsyncMock()
        mock_storage.retrieve.return_value = content

        with (
            patch(
                "backend.copilot.transcript.get_workspace_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
            patch(
                "backend.copilot.transcript._projects_base",
                return_value=projects_base,
            ),
        ):
            result = asyncio.run(
                restore_cli_session(
                    user_id="user-1",
                    session_id=session_id,
                    sdk_cwd=sdk_cwd,
                )
            )

        assert result is True

    def test_returns_false_on_download_exception(self):
        """Non-FileNotFoundError during retrieve logs warning and returns False."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import restore_cli_session

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = RuntimeError("network error")

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                restore_cli_session(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000004",
                    sdk_cwd="/tmp/copilot-test",
                )
            )

        assert result is False
