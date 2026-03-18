"""Tests for retry logic and transcript compaction helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from backend.util import json
from backend.util.prompt import CompressResult

from .conftest import build_test_transcript as _build_transcript
from .service import _friendly_error_text, _is_prompt_too_long
from .transcript import (
    _flatten_assistant_content,
    _flatten_tool_result_content,
    _messages_to_transcript,
    _run_compression,
    _transcript_to_messages,
    compact_transcript,
    validate_transcript,
)

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

    def test_tool_use_blocks(self):
        blocks = [{"type": "tool_use", "name": "read_file", "input": {}}]
        assert _flatten_assistant_content(blocks) == "[tool_use: read_file]"

    def test_mixed_blocks(self):
        blocks = [
            {"type": "text", "text": "Let me read that."},
            {"type": "tool_use", "name": "Read", "input": {"path": "/foo"}},
        ]
        result = _flatten_assistant_content(blocks)
        assert "Let me read that." in result
        assert "[tool_use: Read]" in result

    def test_raw_strings(self):
        assert _flatten_assistant_content(["hello", "world"]) == "hello\nworld"

    def test_unknown_block_type_preserved_as_placeholder(self):
        blocks = [
            {"type": "text", "text": "See this image:"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
        ]
        result = _flatten_assistant_content(blocks)
        assert "See this image:" in result
        assert "[__image__]" in result

    def test_empty(self):
        assert _flatten_assistant_content([]) == ""


# ---------------------------------------------------------------------------
# _flatten_tool_result_content
# ---------------------------------------------------------------------------


class TestFlattenToolResultContent:
    def test_tool_result_with_text(self):
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "123",
                "content": [{"type": "text", "text": "file contents here"}],
            }
        ]
        assert _flatten_tool_result_content(blocks) == "file contents here"

    def test_tool_result_with_string_content(self):
        blocks = [{"type": "tool_result", "tool_use_id": "123", "content": "ok"}]
        assert _flatten_tool_result_content(blocks) == "ok"

    def test_text_block(self):
        blocks = [{"type": "text", "text": "plain text"}]
        assert _flatten_tool_result_content(blocks) == "plain text"

    def test_raw_string(self):
        assert _flatten_tool_result_content(["raw"]) == "raw"

    def test_tool_result_with_none_content(self):
        """tool_result with content=None should produce empty string."""
        blocks = [{"type": "tool_result", "tool_use_id": "x", "content": None}]
        assert _flatten_tool_result_content(blocks) == ""

    def test_tool_result_with_empty_list_content(self):
        """tool_result with content=[] should produce empty string."""
        blocks = [{"type": "tool_result", "tool_use_id": "x", "content": []}]
        assert _flatten_tool_result_content(blocks) == ""

    def test_empty(self):
        assert _flatten_tool_result_content([]) == ""

    def test_nested_dict_without_text(self):
        """Dict blocks without text key use json.dumps fallback."""
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "x",
                "content": [{"type": "image", "source": "data:..."}],
            }
        ]
        result = _flatten_tool_result_content(blocks)
        assert "image" in result  # json.dumps fallback

    def test_unknown_block_type_preserved_as_placeholder(self):
        blocks = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
        result = _flatten_tool_result_content(blocks)
        assert "[__image__]" in result


# ---------------------------------------------------------------------------
# _transcript_to_messages
# ---------------------------------------------------------------------------


def _make_entry(entry_type: str, role: str, content: str | list, **kwargs) -> str:
    """Build a JSONL line for testing."""
    uid = str(uuid4())
    msg: dict = {"role": role, "content": content}
    msg.update(kwargs)
    entry = {
        "type": entry_type,
        "uuid": uid,
        "parentUuid": None,
        "message": msg,
    }
    return json.dumps(entry, separators=(",", ":"))


class TestTranscriptToMessages:
    def test_basic_roundtrip(self):
        lines = [
            _make_entry("user", "user", "Hello"),
            _make_entry("assistant", "assistant", [{"type": "text", "text": "Hi"}]),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}

    def test_skips_strippable_types(self):
        """Progress and metadata entries are excluded."""
        lines = [
            _make_entry("user", "user", "Hello"),
            json.dumps(
                {
                    "type": "progress",
                    "uuid": str(uuid4()),
                    "parentUuid": None,
                    "message": {"role": "assistant", "content": "..."},
                }
            ),
            _make_entry("assistant", "assistant", [{"type": "text", "text": "Hi"}]),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 2

    def test_empty_content(self):
        assert _transcript_to_messages("") == []

    def test_tool_result_content(self):
        """User entries with tool_result content blocks are flattened."""
        lines = [
            _make_entry(
                "user",
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "123",
                        "content": "tool output",
                    }
                ],
            ),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 1
        assert messages[0]["content"] == "tool output"

    def test_malformed_json_lines_skipped(self):
        """Malformed JSON lines in transcript are silently skipped."""
        lines = [
            _make_entry("user", "user", "Hello"),
            "this is not valid json",
            _make_entry("assistant", "assistant", [{"type": "text", "text": "Hi"}]),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 2

    def test_empty_lines_skipped(self):
        """Empty lines and whitespace-only lines are skipped."""
        lines = [
            _make_entry("user", "user", "Hello"),
            "",
            "   ",
            _make_entry("assistant", "assistant", [{"type": "text", "text": "Hi"}]),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 2

    def test_unicode_content_preserved(self):
        """Unicode characters survive transcript roundtrip."""
        lines = [
            _make_entry("user", "user", "Hello 你好 🌍"),
            _make_entry(
                "assistant",
                "assistant",
                [{"type": "text", "text": "Bonjour 日本語 émojis 🎉"}],
            ),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert messages[0]["content"] == "Hello 你好 🌍"
        assert messages[1]["content"] == "Bonjour 日本語 émojis 🎉"

    def test_entry_without_role_skipped(self):
        """Entries with missing role in message are skipped."""
        entry_no_role = json.dumps(
            {
                "type": "user",
                "uuid": str(uuid4()),
                "parentUuid": None,
                "message": {"content": "no role here"},
            }
        )
        lines = [
            entry_no_role,
            _make_entry("user", "user", "Hello"),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

    def test_tool_use_and_result_pairs(self):
        """Tool use + tool result pairs are properly flattened."""
        lines = [
            _make_entry(
                "assistant",
                "assistant",
                [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "name": "read_file", "input": {"path": "/x"}},
                ],
            ),
            _make_entry(
                "user",
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": [{"type": "text", "text": "file contents"}],
                    }
                ],
            ),
        ]
        content = "\n".join(lines) + "\n"
        messages = _transcript_to_messages(content)
        assert len(messages) == 2
        assert "Let me check." in messages[0]["content"]
        assert "[tool_use: read_file]" in messages[0]["content"]
        assert messages[1]["content"] == "file contents"


# ---------------------------------------------------------------------------
# _messages_to_transcript
# ---------------------------------------------------------------------------


class TestMessagesToTranscript:
    def test_produces_valid_jsonl(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = _messages_to_transcript(messages)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "type" in parsed
            assert "uuid" in parsed
            assert "message" in parsed

    def test_assistant_has_proper_structure(self):
        messages = [{"role": "assistant", "content": "Hello"}]
        result = _messages_to_transcript(messages)
        entry = json.loads(result.strip())
        assert entry["type"] == "assistant"
        msg = entry["message"]
        assert msg["role"] == "assistant"
        assert msg["type"] == "message"
        assert msg["stop_reason"] == "end_turn"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"

    def test_user_has_plain_content(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _messages_to_transcript(messages)
        entry = json.loads(result.strip())
        assert entry["type"] == "user"
        assert entry["message"]["content"] == "Hi"

    def test_parent_uuid_chain(self):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = _messages_to_transcript(messages)
        lines = result.strip().split("\n")
        entries = [json.loads(line) for line in lines]
        assert entries[0]["parentUuid"] == ""
        assert entries[1]["parentUuid"] == entries[0]["uuid"]
        assert entries[2]["parentUuid"] == entries[1]["uuid"]

    def test_empty_messages(self):
        assert _messages_to_transcript([]) == ""

    def test_output_is_valid_transcript(self):
        """Output should pass validate_transcript if it has assistant entries."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = _messages_to_transcript(messages)
        assert validate_transcript(result)

    def test_roundtrip_to_messages(self):
        """Messages → transcript → messages preserves structure."""
        original = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        transcript = _messages_to_transcript(original)
        restored = _transcript_to_messages(transcript)
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert orig["role"] == rest["role"]
            assert orig["content"] == rest["content"]


# ---------------------------------------------------------------------------
# compact_transcript
# ---------------------------------------------------------------------------


class TestCompactTranscript:
    @pytest.mark.asyncio
    async def test_too_few_messages_returns_none(self, mock_chat_config):
        """compact_transcript returns None when transcript has < 2 messages."""
        transcript = _build_transcript([("user", "Hello")])
        result = await compact_transcript(transcript, model="test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_compacted(self, mock_chat_config):
        """When compress_context says no compaction needed, returns None.
        The compressor couldn't reduce it, so retrying with the same
        content would fail identically."""
        transcript = _build_transcript(
            [
                ("user", "Hello"),
                ("assistant", "Hi there"),
            ]
        )
        mock_result = type(
            "CompressResult",
            (),
            {
                "was_compacted": False,
                "messages": [],
                "original_token_count": 100,
                "token_count": 100,
                "messages_summarized": 0,
                "messages_dropped": 0,
            },
        )()
        with patch(
            "backend.copilot.sdk.transcript._run_compression",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await compact_transcript(transcript, model="test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_compacted_transcript(self, mock_chat_config):
        """When compaction succeeds, returns a valid compacted transcript."""
        transcript = _build_transcript(
            [
                ("user", "Hello"),
                ("assistant", "Hi"),
                ("user", "More"),
                ("assistant", "Details"),
            ]
        )
        compacted_msgs = [
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "Summarized response"},
        ]
        mock_result = type(
            "CompressResult",
            (),
            {
                "was_compacted": True,
                "messages": compacted_msgs,
                "original_token_count": 500,
                "token_count": 100,
                "messages_summarized": 2,
                "messages_dropped": 0,
            },
        )()
        with patch(
            "backend.copilot.sdk.transcript._run_compression",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await compact_transcript(transcript, model="test-model")
        assert result is not None
        assert validate_transcript(result)
        msgs = _transcript_to_messages(result)
        assert len(msgs) == 2
        assert msgs[1]["content"] == "Summarized response"

    @pytest.mark.asyncio
    async def test_returns_none_on_compression_failure(self, mock_chat_config):
        """When _run_compression raises, returns None."""
        transcript = _build_transcript(
            [
                ("user", "Hello"),
                ("assistant", "Hi"),
            ]
        )
        with patch(
            "backend.copilot.sdk.transcript._run_compression",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = await compact_transcript(transcript, model="test-model")
        assert result is None


# ---------------------------------------------------------------------------
# _is_prompt_too_long
# ---------------------------------------------------------------------------


class TestIsPromptTooLong:
    """Unit tests for _is_prompt_too_long pattern matching."""

    def test_prompt_is_too_long(self):
        err = RuntimeError("prompt is too long for model context")
        assert _is_prompt_too_long(err) is True

    def test_request_too_large(self):
        err = Exception("request too large: 250000 tokens")
        assert _is_prompt_too_long(err) is True

    def test_maximum_context_length(self):
        err = ValueError("maximum context length exceeded")
        assert _is_prompt_too_long(err) is True

    def test_context_length_exceeded(self):
        err = Exception("context_length_exceeded")
        assert _is_prompt_too_long(err) is True

    def test_input_tokens_exceed(self):
        err = Exception("input tokens exceed the max_tokens limit")
        assert _is_prompt_too_long(err) is True

    def test_input_is_too_long(self):
        err = Exception("input is too long for the model")
        assert _is_prompt_too_long(err) is True

    def test_content_length_exceeds(self):
        err = Exception("content length exceeds maximum")
        assert _is_prompt_too_long(err) is True

    def test_unrelated_error_returns_false(self):
        err = RuntimeError("network timeout")
        assert _is_prompt_too_long(err) is False

    def test_auth_error_returns_false(self):
        err = Exception("authentication failed: invalid API key")
        assert _is_prompt_too_long(err) is False

    def test_chained_exception_detected(self):
        """Prompt-too-long error wrapped in another exception is detected."""
        inner = RuntimeError("prompt is too long")
        outer = Exception("SDK error")
        outer.__cause__ = inner
        assert _is_prompt_too_long(outer) is True

    def test_case_insensitive(self):
        err = Exception("PROMPT IS TOO LONG")
        assert _is_prompt_too_long(err) is True

    def test_old_max_tokens_exceeded_not_matched(self):
        """The old broad 'max_tokens_exceeded' pattern was removed.
        Only 'input tokens exceed' should match now."""
        err = Exception("max_tokens_exceeded")
        assert _is_prompt_too_long(err) is False


# ---------------------------------------------------------------------------
# _run_compression timeout fallback
# ---------------------------------------------------------------------------


class TestRunCompressionTimeout:
    """Verify _run_compression falls back to truncation when LLM times out."""

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_truncation(self):
        """When compress_context with LLM client times out,
        _run_compression falls back to truncation (client=None)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        truncation_result = CompressResult(
            messages=messages,
            was_compacted=False,
            original_token_count=50,
            token_count=50,
            messages_summarized=0,
            messages_dropped=0,
        )

        call_args: list[dict] = []

        async def _mock_compress(**kwargs):
            call_args.append(kwargs)
            if kwargs.get("client") is not None:
                # Simulate timeout by raising asyncio.TimeoutError
                raise asyncio.TimeoutError("LLM compaction timed out")
            return truncation_result

        with (
            patch(
                "backend.copilot.sdk.transcript.get_openai_client",
                return_value="fake-client",
            ),
            patch(
                "backend.copilot.sdk.transcript.compress_context",
                side_effect=_mock_compress,
            ),
        ):
            result = await _run_compression(messages, "test-model", "[test]")

        assert result == truncation_result
        # Should have been called twice: once with client, once without
        assert len(call_args) == 2
        assert call_args[0]["client"] is not None  # LLM attempt
        assert call_args[1]["client"] is None  # truncation fallback

    @pytest.mark.asyncio
    async def test_no_client_uses_truncation_directly(self):
        """When no OpenAI client is configured, goes straight to truncation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        truncation_result = CompressResult(
            messages=messages,
            was_compacted=False,
            original_token_count=50,
            token_count=50,
            messages_summarized=0,
            messages_dropped=0,
        )

        with (
            patch(
                "backend.copilot.sdk.transcript.get_openai_client",
                return_value=None,
            ),
            patch(
                "backend.copilot.sdk.transcript.compress_context",
                new_callable=AsyncMock,
                return_value=truncation_result,
            ) as mock_compress,
        ):
            result = await _run_compression(messages, "test-model", "[test]")

        assert result == truncation_result
        mock_compress.assert_called_once()
        # When no client, compress_context is called with client=None
        assert mock_compress.call_args.kwargs.get("client") is None


# ---------------------------------------------------------------------------
# _friendly_error_text
# ---------------------------------------------------------------------------


class TestFriendlyErrorText:
    """Verify user-friendly error message mapping."""

    def test_authentication_error(self):
        result = _friendly_error_text("authentication failed: invalid API key")
        assert "Authentication" in result
        assert "API key" in result

    def test_rate_limit_error(self):
        result = _friendly_error_text("rate limit exceeded")
        assert "Rate limit" in result

    def test_overloaded_error(self):
        result = _friendly_error_text("API is overloaded")
        assert "overloaded" in result

    def test_timeout_error(self):
        result = _friendly_error_text("Request timeout after 30s")
        assert "timed out" in result

    def test_connection_error(self):
        result = _friendly_error_text("Connection refused")
        assert "Connection" in result or "connection" in result

    def test_unknown_error_passthrough(self):
        result = _friendly_error_text("some unknown error XYZ")
        assert "SDK stream error:" in result
        assert "XYZ" in result

    def test_unauthorized_error(self):
        result = _friendly_error_text("401 Unauthorized")
        assert "Authentication" in result
