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
    _rechain_tail,
    _sanitize_id,
    _transcript_to_messages,
    detect_gap,
    extract_context_messages,
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
                    # Anthropic-style thinking block — has a signature so
                    # ``_should_strip_thinking_block`` preserves it on the
                    # last turn.  Without the signature (e.g. emitted by
                    # Kimi K2.6 via OpenRouter) it would be stripped — see
                    # ``test_strips_signatureless_thinking_from_last_turn``.
                    {
                        "type": "thinking",
                        "thinking": "fresh thinking",
                        "signature": "anthropic-signed-blob",
                    },
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

    def test_strips_signatureless_thinking_from_last_turn(self):
        """Kimi K2.6 (and other non-Anthropic OpenRouter providers) emit
        thinking blocks without the Anthropic ``signature`` field.  When
        a subsequent advanced-tier toggle replays the transcript to Opus,
        Anthropic's API rejects the signature-less block with ``Invalid
        `signature` in `thinking` block`` — so strip_for_upload must drop
        them from the LAST assistant entry too, not just stale ones."""
        user = {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "",
            "message": {"role": "user", "content": "hi"},
        }
        # Last (and only) assistant entry with a Kimi-shape thinking block
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {
                "role": "assistant",
                "id": "msg_kimi",
                "content": [
                    # No ``signature`` field → non-Anthropic provider
                    {"type": "thinking", "thinking": "kimi reasoning"},
                    {"type": "text", "text": "answer"},
                ],
            },
        }
        content = _make_jsonl(user, asst)
        result = strip_for_upload(content)
        entries = [json.loads(line) for line in result.strip().split("\n")]
        asst_entry = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_kimi"
        )
        types = [
            b["type"] for b in asst_entry["message"]["content"] if isinstance(b, dict)
        ]
        assert "thinking" not in types, (
            "Signature-less thinking block on last turn must be stripped "
            "to prevent Anthropic API rejection on model-switch replay"
        )
        assert "text" in types, "Text content must survive stripping"

    def test_strips_non_anthropic_thinking_with_placeholder_signature(self):
        """OpenRouter's Anthropic-compat shim can emit thinking blocks
        from non-Anthropic producers (Kimi K2.6, DeepSeek) with a
        PLACEHOLDER signature string that passes the "non-empty string"
        check but fails Anthropic's cryptographic validation on replay.

        Observed in session 864a55ba after model-toggle from standard
        (Kimi) to advanced (Opus): the CLI session upload included a
        thinking block with ``signature="ANTHROPIC_SHIM_PLACEHOLDER"``
        (or similar), Opus 4.7 rejected with 400 ``Invalid `signature`
        in `thinking` block``.  Fix: strip thinking blocks from the
        LAST assistant turn whenever the producing ``model`` isn't an
        ``anthropic/*`` slug, regardless of signature presence."""
        user = {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "",
            "message": {"role": "user", "content": "hi"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {
                "role": "assistant",
                "id": "msg_kimi_shim",
                "model": "moonshotai/kimi-k2.6-20260420",
                "content": [
                    # Placeholder signature — non-empty but cryptographically
                    # invalid for Anthropic.  Legacy strip (signature-only)
                    # would KEEP this block.
                    {
                        "type": "thinking",
                        "thinking": "shimmed reasoning",
                        "signature": "PLACEHOLDER_SHIM_SIG_abc123",
                    },
                    {"type": "text", "text": "answer"},
                ],
            },
        }
        content = _make_jsonl(user, asst)
        result = strip_for_upload(content)
        entries = [json.loads(line) for line in result.strip().split("\n")]
        asst_entry = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_kimi_shim"
        )
        types = [
            b["type"] for b in asst_entry["message"]["content"] if isinstance(b, dict)
        ]
        assert "thinking" not in types, (
            "Non-Anthropic thinking block must be stripped even when it "
            "carries a placeholder signature — replay-to-Opus otherwise "
            "400s with Invalid signature"
        )
        assert "text" in types

    def test_preserves_anthropic_thinking_on_non_last_turn(self):
        """Anthropic ``thinking`` blocks on NON-last turns carry real
        reasoning state that helps context continuity on ``--resume``.
        Keep them when the producing model is known-Anthropic with a
        valid signature; strip only when we can't validate safely
        (legacy callers with no model info — falls through to the
        old stale-strip rule).
        """
        user = {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "",
            "message": {"role": "user", "content": "first"},
        }
        asst1 = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {
                "role": "assistant",
                "id": "msg_opus_prev",
                "model": "anthropic/claude-4.7-opus-20260416",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "first-turn reasoning",
                        "signature": "ANTHROPIC_SIG_1",
                    },
                    {"type": "text", "text": "first answer"},
                ],
            },
        }
        user2 = {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "message": {"role": "user", "content": "second"},
        }
        asst2 = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "message": {
                "role": "assistant",
                "id": "msg_opus_last",
                "model": "anthropic/claude-4.7-opus-20260416",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "last-turn reasoning",
                        "signature": "ANTHROPIC_SIG_2",
                    },
                    {"type": "text", "text": "last answer"},
                ],
            },
        }
        content = _make_jsonl(user, asst1, user2, asst2)
        result = strip_for_upload(content)
        entries = [json.loads(line) for line in result.strip().split("\n")]

        # Prior Opus turn's thinking must survive — valid Anthropic
        # block with signature.
        prev = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_opus_prev"
        )
        prev_types = [b["type"] for b in prev["message"]["content"]]
        assert "thinking" in prev_types, (
            "Anthropic thinking block on a non-last turn must be "
            "preserved — it carries real reasoning state"
        )
        # Last turn's thinking also preserved.
        last = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_opus_last"
        )
        last_types = [b["type"] for b in last["message"]["content"]]
        assert "thinking" in last_types

    def test_preserves_anthropic_thinking_with_valid_signature(self):
        """Sanity: an Anthropic-issued thinking block with a real
        signature on the last turn must NOT be stripped — Anthropic
        requires value-identity on replay."""
        user = {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "",
            "message": {"role": "user", "content": "hi"},
        }
        asst = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {
                "role": "assistant",
                "id": "msg_opus",
                "model": "anthropic/claude-4.7-opus-20260416",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "reasoning",
                        "signature": "REAL_ANTHROPIC_SIG_blob",
                    },
                    {"type": "text", "text": "answer"},
                ],
            },
        }
        content = _make_jsonl(user, asst)
        result = strip_for_upload(content)
        entries = [json.loads(line) for line in result.strip().split("\n")]
        asst_entry = next(
            e for e in entries if e.get("message", {}).get("id") == "msg_opus"
        )
        types = [
            b["type"] for b in asst_entry["message"]["content"] if isinstance(b, dict)
        ]
        assert (
            "thinking" in types
        ), "Anthropic-signed thinking on last turn must survive strip"
        assert "text" in types

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
        from .transcript import cli_session_path, projects_base

        sdk_cwd = "/tmp/copilot-abc"
        result = cli_session_path(sdk_cwd, "12345678-1234-1234-1234-123456789abc")
        base = projects_base()
        assert result.startswith(base)
        # Encoded cwd replaces '/' with '-'
        assert "-tmp-copilot-abc" in result
        assert result.endswith(".jsonl")

    def test_sanitizes_session_id(self):
        from .transcript import cli_session_path

        result = cli_session_path("/tmp/cwd", "../../etc/passwd")
        # _sanitize_id strips non-hex/hyphen chars; path traversal impossible
        assert ".." not in result
        assert "passwd" not in result


class TestUploadCliSession:
    def test_uploads_content_bytes_successfully(self):
        """Happy path: content bytes are stored as jsonl + meta.json."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_transcript

        mock_storage = AsyncMock()
        content = b'{"type":"assistant"}\n'

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            asyncio.run(
                upload_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000001",
                    content=content,
                )
            )

        # Two calls expected: session JSONL + companion .meta.json
        assert mock_storage.store.call_count == 2

    def test_uploads_companion_meta_json_with_message_count(self):
        """upload_transcript stores a companion .meta.json with message_count."""
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_transcript

        mock_storage = AsyncMock()
        content = b'{"type":"assistant"}\n'

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            asyncio.run(
                upload_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000010",
                    content=content,
                    message_count=5,
                )
            )

        assert mock_storage.store.call_count == 2
        # Find the meta.json store call
        meta_call = next(
            c
            for c in mock_storage.store.call_args_list
            if c.kwargs.get("filename", "").endswith(".meta.json")
        )
        meta_content = json.loads(meta_call.kwargs["content"])
        assert meta_content["message_count"] == 5

    def test_skips_upload_on_storage_failure(self):
        """Storage exception on jsonl write is logged and does not propagate.

        With sequential writes, JSONL failure returns early — meta store is
        never called, so no rollback is needed.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_transcript

        mock_storage = AsyncMock()
        mock_storage.store.side_effect = RuntimeError("gcs unavailable")
        content = b'{"type":"assistant"}\n'

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            # Should not raise — failures are logged as warnings
            asyncio.run(
                upload_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000002",
                    content=content,
                )
            )

        # Only one store call attempted (the JSONL); meta never reached
        mock_storage.store.assert_called_once()
        mock_storage.delete.assert_not_called()

    def test_rolls_back_session_when_meta_upload_fails(self):
        """When meta upload fails after JSONL succeeds, JSONL is rolled back.

        Guarantees the pair is either both present or both absent — avoids an
        orphaned JSONL being used with wrong mode/watermark defaults.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_transcript

        mock_storage = AsyncMock()
        # First store (JSONL) succeeds; second store (meta) fails
        mock_storage.store.side_effect = [None, RuntimeError("meta write failed")]
        content = b'{"type":"assistant"}\n'

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            asyncio.run(
                upload_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000099",
                    content=content,
                )
            )

        # Both store calls were attempted (JSONL then meta)
        assert mock_storage.store.call_count == 2
        # JSONL should be rolled back via delete
        mock_storage.delete.assert_called_once()

    def test_baseline_mode_stored_in_meta(self):
        """upload_transcript with mode='baseline' stores mode in companion meta.json."""
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch

        from .transcript import upload_transcript

        mock_storage = AsyncMock()
        content = b'{"type":"assistant"}\n'

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            asyncio.run(
                upload_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000098",
                    content=content,
                    message_count=4,
                    mode="baseline",
                )
            )

        meta_call = next(
            c
            for c in mock_storage.store.call_args_list
            if c.kwargs.get("filename", "").endswith(".meta.json")
        )
        meta_content = json.loads(meta_call.kwargs["content"])
        assert meta_content["mode"] == "baseline"
        assert meta_content["message_count"] == 4

    def test_strips_session_before_upload_and_writes_back(self):
        """strip_for_upload removes progress entries and returns smaller content."""
        import json

        from .transcript import strip_for_upload

        progress_entry = {
            "type": "progress",
            "uuid": "p1",
            "parentUuid": "u1",
            "data": {"type": "bash_progress", "stdout": "running..."},
        }
        user_entry = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "hello"},
        }
        asst_entry = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {"role": "assistant", "content": "world"},
        }
        raw_content = (
            json.dumps(progress_entry)
            + "\n"
            + json.dumps(user_entry)
            + "\n"
            + json.dumps(asst_entry)
            + "\n"
        )

        stripped = strip_for_upload(raw_content)

        stored_lines = stripped.strip().split("\n")
        stored_types = [json.loads(line).get("type") for line in stored_lines]
        assert "progress" not in stored_types
        assert "user" in stored_types
        assert "assistant" in stored_types
        assert len(stripped.encode()) < len(raw_content.encode())

    def test_strips_stale_thinking_blocks_before_upload(self):
        """strip_for_upload removes thinking blocks from non-last assistant turns."""
        import json

        from .transcript import strip_for_upload

        u1 = {
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "q1"},
        }
        a1_with_thinking = {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {
                "id": "msg_a1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "A" * 5000},
                    {"type": "text", "text": "answer1"},
                ],
            },
        }
        u2 = {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "message": {"role": "user", "content": "q2"},
        }
        a2_no_thinking = {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "message": {
                "id": "msg_a2",
                "role": "assistant",
                "content": [{"type": "text", "text": "answer2"}],
            },
        }
        raw_content = (
            json.dumps(u1)
            + "\n"
            + json.dumps(a1_with_thinking)
            + "\n"
            + json.dumps(u2)
            + "\n"
            + json.dumps(a2_no_thinking)
            + "\n"
        )

        stripped = strip_for_upload(raw_content)

        stored_lines = stripped.strip().split("\n")

        # a1 should have its thinking block stripped (it's not the last assistant turn).
        a1_stored = json.loads(stored_lines[1])
        a1_content = a1_stored["message"]["content"]
        assert all(
            b["type"] != "thinking" for b in a1_content
        ), "stale thinking block should be stripped from a1"
        assert any(
            b["type"] == "text" for b in a1_content
        ), "text block should be kept in a1"

        # a2 (last turn) should be unchanged.
        a2_stored = json.loads(stored_lines[3])
        assert a2_stored["message"]["content"] == [{"type": "text", "text": "answer2"}]


class TestRestoreCliSession:
    def test_returns_none_when_file_not_found_in_storage(self):
        """Returns None (graceful degradation) when the session is missing."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [
            FileNotFoundError("no session"),
            FileNotFoundError("no meta"),
        ]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000000",
                )
            )

        assert result is None

    def test_returns_transcript_download_on_success_no_meta(self):
        """Happy path with no meta.json: returns TranscriptDownload with message_count=0."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        session_id = "12345678-0000-0000-0000-000000000003"
        content = b'{"type":"assistant"}\n'

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [content, FileNotFoundError("no meta")]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id=session_id,
                )
            )

        assert isinstance(result, TranscriptDownload)
        assert result.content == content
        assert result.message_count == 0
        assert result.mode == "sdk"

    def test_returns_transcript_download_with_message_count_from_meta(self):
        """When meta.json is present, message_count and mode are read from it."""
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        session_id = "12345678-0000-0000-0000-000000000005"
        content = b'{"type":"assistant"}\n'
        meta_bytes = json.dumps(
            {"message_count": 7, "mode": "sdk", "uploaded_at": 1234567.0}
        ).encode()

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [content, meta_bytes]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id=session_id,
                )
            )

        assert isinstance(result, TranscriptDownload)
        assert result.content == content
        assert result.message_count == 7
        assert result.mode == "sdk"

    def test_returns_none_on_download_exception(self):
        """Non-FileNotFoundError during retrieve logs warning and returns None."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [
            RuntimeError("network error"),
            FileNotFoundError("no meta"),
        ]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000004",
                )
            )

        assert result is None

    def test_baseline_mode_in_meta_returned(self):
        """When meta.json contains mode='baseline', result.mode is 'baseline'."""
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        content = b'{"type":"assistant"}\n'
        meta_bytes = json.dumps(
            {"message_count": 3, "mode": "baseline", "uploaded_at": 0.0}
        ).encode()

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [content, meta_bytes]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000020",
                )
            )

        assert isinstance(result, TranscriptDownload)
        assert result.mode == "baseline"
        assert result.message_count == 3

    def test_invalid_mode_in_meta_defaults_to_sdk(self):
        """Unknown mode value in meta.json falls back to 'sdk'."""
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        content = b'{"type":"assistant"}\n'
        meta_bytes = json.dumps({"message_count": 2, "mode": "unknown_mode"}).encode()

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [content, meta_bytes]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000021",
                )
            )

        assert isinstance(result, TranscriptDownload)
        assert result.mode == "sdk"

    def test_invalid_utf8_meta_uses_defaults(self):
        """Meta bytes that fail UTF-8 decode fall back to message_count=0, mode='sdk'."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        content = b'{"type":"assistant"}\n'
        bad_meta = b"\xff\xfe"

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [content, bad_meta]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000022",
                )
            )

        assert isinstance(result, TranscriptDownload)
        assert result.message_count == 0
        assert result.mode == "sdk"

    def test_meta_fetch_exception_uses_defaults(self):
        """Non-FileNotFoundError on meta fetch still returns content with defaults."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        from .transcript import download_transcript

        content = b'{"type":"assistant"}\n'

        mock_storage = AsyncMock()
        mock_storage.retrieve.side_effect = [content, RuntimeError("meta unavailable")]

        with patch(
            "backend.copilot.transcript.get_workspace_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            result = asyncio.run(
                download_transcript(
                    user_id="user-1",
                    session_id="12345678-0000-0000-0000-000000000023",
                )
            )

        assert isinstance(result, TranscriptDownload)
        assert result.content == content
        assert result.message_count == 0
        assert result.mode == "sdk"


# ---------------------------------------------------------------------------
# detect_gap
# ---------------------------------------------------------------------------


def _msgs(*roles: str):
    """Build a list of ChatMessage objects with the given roles."""
    from .model import ChatMessage

    return [ChatMessage(role=r, content=f"{r}-{i}") for i, r in enumerate(roles)]


class TestDetectGap:
    """``detect_gap`` returns messages between transcript watermark and current turn."""

    def _dl(self, message_count: int) -> TranscriptDownload:
        return TranscriptDownload(content=b"", message_count=message_count, mode="sdk")

    def test_zero_watermark_returns_empty(self):
        """message_count=0 means no watermark — skip gap detection."""
        dl = self._dl(0)
        messages = _msgs("user", "assistant", "user")
        assert detect_gap(dl, messages) == []

    def test_watermark_covers_all_prefix_returns_empty(self):
        """Transcript already covers all messages up to the current user turn."""
        # session: [user, assistant, user(current)] — wm=2 means covers up to assistant
        dl = self._dl(2)
        messages = _msgs("user", "assistant", "user")
        assert detect_gap(dl, messages) == []

    def test_watermark_exceeds_session_returns_empty(self):
        """Watermark ahead of session count (race / over-count) → no gap."""
        dl = self._dl(10)
        messages = _msgs("user", "assistant", "user")
        assert detect_gap(dl, messages) == []

    def test_misaligned_watermark_not_on_assistant_returns_empty(self):
        """Watermark at a user-role position is misaligned — skip gap."""
        # wm=1: position 0 is 'user', not 'assistant' → skip
        dl = self._dl(1)
        messages = _msgs("user", "assistant", "user", "assistant", "user")
        assert detect_gap(dl, messages) == []

    def test_returns_gap_messages(self):
        """Watermark behind session — gap messages returned (excluding current turn)."""
        # session: [user0, assistant1, user2, assistant3, user4(current)]
        # wm=2: transcript covers [0,1]; gap = [user2, assistant3]
        dl = self._dl(2)
        messages = _msgs("user", "assistant", "user", "assistant", "user")
        gap = detect_gap(dl, messages)
        assert len(gap) == 2
        assert gap[0].role == "user"
        assert gap[1].role == "assistant"

    def test_excludes_current_user_turn(self):
        """The last message (current user turn) is never included in the gap."""
        # wm=2, session has 4 msgs: gap = [msg2] only (msg3 is current turn → excluded)
        dl = self._dl(2)
        messages = _msgs("user", "assistant", "user", "user")
        gap = detect_gap(dl, messages)
        assert len(gap) == 1
        assert gap[0].role == "user"

    def test_single_gap_message(self):
        """One message between watermark and current turn."""
        # session: [user0, assistant1, user2, assistant3, user4(current)]
        # wm=3: position 2 is 'user' → misaligned, returns []
        # use wm=4: but 4 >= total-1=4 → also empty
        # wm=3 with session [u, a, u, a, u, a, u(current)]: position 2 is 'user' → empty
        # Valid case: wm=2 has 3 messages (assistant at 1), wm=4 with [u,a,u,a,u,a,u]:
        # let's use wm=4 with 7 messages: wm=4 >= total-1=6? no, 4<6. pos[3]=assistant → gap=[msg4,msg5]
        # simpler: wm=2, [u0,a1,a2,u3(current)] — pos[1]=assistant, gap=[a2] only
        dl = self._dl(2)
        messages = _msgs("user", "assistant", "assistant", "user")
        gap = detect_gap(dl, messages)
        assert len(gap) == 1
        assert gap[0].role == "assistant"


# ---------------------------------------------------------------------------
# extract_context_messages
# ---------------------------------------------------------------------------


def _make_valid_transcript(*roles: str) -> str:
    """Build a minimal valid JSONL transcript with the given message roles."""
    import json as stdlib_json

    from .transcript import STOP_REASON_END_TURN

    lines = []
    parent = ""
    for i, role in enumerate(roles):
        uid = f"uid-{i}"
        entry: dict = {
            "type": role,
            "uuid": uid,
            "parentUuid": parent,
            "message": {
                "role": role,
                "content": f"{role} content {i}",
            },
        }
        if role == "assistant":
            entry["message"]["id"] = f"msg_{i}"
            entry["message"]["model"] = "test-model"
            entry["message"]["type"] = "message"
            entry["message"]["stop_reason"] = STOP_REASON_END_TURN
            entry["message"]["content"] = [
                {"type": "text", "text": f"assistant content {i}"}
            ]
        lines.append(stdlib_json.dumps(entry))
        parent = uid
    return "\n".join(lines) + "\n"


class TestExtractContextMessages:
    """``extract_context_messages`` returns the shared context primitive."""

    def test_none_download_returns_prior(self):
        """No download → falls back to all session messages except current turn."""
        messages = _msgs("user", "assistant", "user")
        result = extract_context_messages(None, messages)
        assert result == messages[:-1]
        assert len(result) == 2

    def test_empty_content_download_returns_prior(self):
        """Empty bytes content → falls back to all prior messages."""
        dl = TranscriptDownload(content=b"", message_count=2, mode="sdk")
        messages = _msgs("user", "assistant", "user")
        result = extract_context_messages(dl, messages)
        assert result == messages[:-1]

    def test_valid_transcript_no_gap_returns_transcript_messages(self):
        """Transcript covers all prior turns → only transcript messages returned."""
        # Transcript: [user, assistant] — 2 messages
        # Session: [user, assistant, user(current)] — watermark=2 covers prefix
        transcript_content = _make_valid_transcript("user", "assistant")
        dl = TranscriptDownload(
            content=transcript_content.encode("utf-8"), message_count=2, mode="sdk"
        )
        messages = _msgs("user", "assistant", "user")
        result = extract_context_messages(dl, messages)
        # Transcript has 2 messages (user + assistant) and no gap
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"

    def test_valid_transcript_with_gap_returns_transcript_plus_gap(self):
        """Transcript is stale → gap messages appended after transcript content."""
        # Transcript: [user, assistant] — watermark=2
        # Session: [user, assistant, user, assistant, user(current)]
        # Gap: [user(2), assistant(3)] — positions 2 and 3
        transcript_content = _make_valid_transcript("user", "assistant")
        dl = TranscriptDownload(
            content=transcript_content.encode("utf-8"), message_count=2, mode="sdk"
        )
        messages = _msgs("user", "assistant", "user", "assistant", "user")
        result = extract_context_messages(dl, messages)
        # 2 transcript messages + 2 gap messages = 4
        assert len(result) == 4
        assert result[0].role == "user"  # transcript user
        assert result[1].role == "assistant"  # transcript assistant
        assert result[2].role == "user"  # gap user
        assert result[3].role == "assistant"  # gap assistant

    def test_compact_summary_entries_preserved(self):
        """``isCompactSummary=True`` entries survive ``_transcript_to_messages``."""
        import json as stdlib_json

        from .transcript import STOP_REASON_END_TURN

        # Build a transcript where one entry is a compaction summary.
        # isCompactSummary=True entries have type in STRIPPABLE_TYPES but are kept.
        compact_entry = stdlib_json.dumps(
            {
                "type": "summary",
                "uuid": "uid-compact",
                "parentUuid": "",
                "isCompactSummary": True,
                "message": {
                    "role": "user",
                    "content": "COMPACT_SUMMARY_CONTENT",
                },
            }
        )
        assistant_entry = stdlib_json.dumps(
            {
                "type": "assistant",
                "uuid": "uid-1",
                "parentUuid": "uid-compact",
                "message": {
                    "role": "assistant",
                    "id": "msg_1",
                    "model": "test",
                    "type": "message",
                    "stop_reason": STOP_REASON_END_TURN,
                    "content": [{"type": "text", "text": "response after compact"}],
                },
            }
        )
        content = compact_entry + "\n" + assistant_entry + "\n"
        dl = TranscriptDownload(
            content=content.encode("utf-8"), message_count=2, mode="sdk"
        )
        messages = _msgs("user", "assistant", "user")
        result = extract_context_messages(dl, messages)
        # Both the compact summary and the assistant response are present
        assert len(result) == 2
        roles = [m.role for m in result]
        assert "user" in roles  # compact summary has role=user
        assert "assistant" in roles
        # The compact summary content is preserved
        compact_msgs = [m for m in result if m.role == "user"]
        assert any("COMPACT_SUMMARY_CONTENT" in (m.content or "") for m in compact_msgs)
