"""Tests for thinking/redacted_thinking block preservation.

Validates the fix for the Anthropic API error:
    "thinking or redacted_thinking blocks in the latest assistant message
    cannot be modified. These blocks must remain as they were in the
    original response."

The API requires that thinking blocks in the LAST assistant message are
preserved byte-for-byte. Older assistant messages may have thinking blocks
stripped entirely. This test suite covers:

  1. _flatten_assistant_content — strips thinking from older messages
  2. compact_transcript — preserves last assistant's thinking blocks
  3. response_adapter — handles ThinkingBlock without error
  4. _format_sdk_content_blocks — preserves redacted_thinking blocks
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock, ThinkingBlock

from backend.copilot.response_model import (
    StreamStartStep,
    StreamTextDelta,
    StreamTextStart,
)
from backend.util import json

from .conftest import build_structured_transcript
from .response_adapter import SDKResponseAdapter
from .service import _format_sdk_content_blocks
from .transcript import (
    _find_last_assistant_entry,
    _flatten_assistant_content,
    _transcript_to_messages,
    compact_transcript,
    validate_transcript,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic thinking block content
# ---------------------------------------------------------------------------

THINKING_BLOCK = {
    "type": "thinking",
    "thinking": "Let me analyze the user's request carefully...",
    "signature": "ErUBCkYIAxgCIkD0V2MsRXPkuGolGexaW9V1kluijxXGF",
}

REDACTED_THINKING_BLOCK = {
    "type": "redacted_thinking",
    "data": "EmwKAhgBEgy2VEE8PJaS2oLJCPkaT...",
}


def _make_thinking_transcript() -> str:
    """Build a transcript with thinking blocks in multiple assistant turns.

    Layout:
      User 1 → Assistant 1 (thinking + text + tool_use)
      User 2 (tool_result) → Assistant 2 (thinking + text)
      User 3 → Assistant 3 (thinking + redacted_thinking + text) ← LAST
    """
    return build_structured_transcript(
        [
            ("user", "What files are in this project?"),
            (
                "assistant",
                [
                    {
                        "type": "thinking",
                        "thinking": "I should list the files.",
                        "signature": "sig_old_1",
                    },
                    {"type": "text", "text": "Let me check the files."},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "list_files",
                        "input": {"path": "/"},
                    },
                ],
            ),
            ("user", "Here are the files: a.py, b.py"),
            (
                "assistant",
                [
                    {
                        "type": "thinking",
                        "thinking": "Good, I see two Python files.",
                        "signature": "sig_old_2",
                    },
                    {"type": "text", "text": "I found a.py and b.py."},
                ],
            ),
            ("user", "Tell me about a.py"),
            (
                "assistant",
                [
                    THINKING_BLOCK,
                    REDACTED_THINKING_BLOCK,
                    {"type": "text", "text": "a.py contains the main entry point."},
                ],
            ),
        ]
    )


def _last_assistant_content(transcript_jsonl: str) -> list[dict] | None:
    """Extract the content blocks of the last assistant entry in a transcript."""
    last_content = None
    for line in transcript_jsonl.strip().split("\n"):
        entry = json.loads(line)
        msg = entry.get("message", {})
        if msg.get("role") == "assistant":
            last_content = msg.get("content")
    return last_content


# ---------------------------------------------------------------------------
# _find_last_assistant_entry — unit tests
# ---------------------------------------------------------------------------


class TestFindLastAssistantEntry:
    def test_splits_at_last_assistant(self):
        """Prefix contains everything before last assistant; tail starts at it."""
        transcript = build_structured_transcript(
            [
                ("user", "Hello"),
                ("assistant", [{"type": "text", "text": "Hi"}]),
                ("user", "More"),
                ("assistant", [{"type": "text", "text": "Details"}]),
            ]
        )
        prefix, tail = _find_last_assistant_entry(transcript)
        # 3 entries in prefix (user, assistant, user), 1 in tail (last assistant)
        assert len(prefix) == 3
        assert len(tail) == 1

    def test_no_assistant_returns_all_in_prefix(self):
        """When there's no assistant, all lines are in prefix, tail is empty."""
        transcript = build_structured_transcript(
            [("user", "Hello"), ("user", "Another question")]
        )
        prefix, tail = _find_last_assistant_entry(transcript)
        assert len(prefix) == 2
        assert tail == []

    def test_assistant_at_index_zero(self):
        """When assistant is the first entry, prefix is empty."""
        transcript = build_structured_transcript(
            [("assistant", [{"type": "text", "text": "Start"}])]
        )
        prefix, tail = _find_last_assistant_entry(transcript)
        assert prefix == []
        assert len(tail) == 1

    def test_trailing_user_included_in_tail(self):
        """User message after last assistant is part of the tail."""
        transcript = build_structured_transcript(
            [
                ("user", "Q1"),
                ("assistant", [{"type": "text", "text": "A1"}]),
                ("user", "Q2"),
            ]
        )
        prefix, tail = _find_last_assistant_entry(transcript)
        assert len(prefix) == 1  # first user
        assert len(tail) == 2  # last assistant + trailing user


# ---------------------------------------------------------------------------
# _flatten_assistant_content — thinking blocks
# ---------------------------------------------------------------------------


class TestFlattenThinkingBlocks:
    def test_thinking_blocks_are_stripped(self):
        """Thinking blocks should not appear in flattened text for compression."""
        blocks = [
            {"type": "thinking", "thinking": "secret thoughts", "signature": "sig"},
            {"type": "text", "text": "Hello user"},
        ]
        result = _flatten_assistant_content(blocks)
        assert "secret thoughts" not in result
        assert "Hello user" in result

    def test_redacted_thinking_blocks_are_stripped(self):
        """Redacted thinking blocks should not appear in flattened text."""
        blocks = [
            {"type": "redacted_thinking", "data": "encrypted_data"},
            {"type": "text", "text": "Response text"},
        ]
        result = _flatten_assistant_content(blocks)
        assert "encrypted_data" not in result
        assert "Response text" in result

    def test_thinking_only_message_flattens_to_empty(self):
        """A message with only thinking blocks flattens to empty string."""
        blocks = [
            {"type": "thinking", "thinking": "just thinking...", "signature": "sig"},
        ]
        result = _flatten_assistant_content(blocks)
        assert result == ""

    def test_mixed_thinking_text_tool(self):
        """Mixed blocks: only text and tool_use survive flattening."""
        blocks = [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "redacted_thinking", "data": "xyz"},
            {"type": "text", "text": "I'll read the file."},
            {"type": "tool_use", "name": "Read", "input": {"path": "/x"}},
        ]
        result = _flatten_assistant_content(blocks)
        assert "hmm" not in result
        assert "xyz" not in result
        assert "I'll read the file." in result
        assert "[tool_use: Read]" in result


# ---------------------------------------------------------------------------
# compact_transcript — thinking block preservation
# ---------------------------------------------------------------------------


class TestCompactTranscriptThinkingBlocks:
    """Verify that compact_transcript preserves thinking blocks in the
    last assistant message while stripping them from older messages."""

    @pytest.mark.asyncio
    async def test_last_assistant_thinking_blocks_preserved(self, mock_chat_config):
        """After compaction, the last assistant entry must retain its
        original thinking and redacted_thinking blocks verbatim."""
        transcript = _make_thinking_transcript()

        compacted_msgs = [
            {"role": "user", "content": "[conversation summary]"},
            {"role": "assistant", "content": "Summarized response"},
        ]
        mock_result = type(
            "CompressResult",
            (),
            {
                "was_compacted": True,
                "messages": compacted_msgs,
                "original_token_count": 800,
                "token_count": 200,
                "messages_summarized": 4,
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

        last_content = _last_assistant_content(result)
        assert last_content is not None, "No assistant entry found"
        assert isinstance(last_content, list)

        # The last assistant must have the thinking blocks preserved
        block_types = [b["type"] for b in last_content]
        assert (
            "thinking" in block_types
        ), "thinking block missing from last assistant message"
        assert (
            "redacted_thinking" in block_types
        ), "redacted_thinking block missing from last assistant message"
        assert "text" in block_types

        # Verify the thinking block content is byte-for-byte identical
        thinking_blocks = [b for b in last_content if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == THINKING_BLOCK["thinking"]
        assert thinking_blocks[0]["signature"] == THINKING_BLOCK["signature"]

        redacted_blocks = [b for b in last_content if b["type"] == "redacted_thinking"]
        assert len(redacted_blocks) == 1
        assert redacted_blocks[0]["data"] == REDACTED_THINKING_BLOCK["data"]

    @pytest.mark.asyncio
    async def test_older_assistant_thinking_blocks_stripped(self, mock_chat_config):
        """Older assistant messages should NOT retain thinking blocks
        after compaction (they're compressed into summaries)."""
        transcript = _make_thinking_transcript()

        # The compressor will receive messages where older assistant
        # entries have already had thinking blocks stripped.
        captured_messages: list[dict] = []

        async def mock_compression(messages, model, log_prefix):
            captured_messages.extend(messages)
            return type(
                "CompressResult",
                (),
                {
                    "was_compacted": True,
                    "messages": messages,
                    "original_token_count": 800,
                    "token_count": 400,
                    "messages_summarized": 2,
                    "messages_dropped": 0,
                },
            )()

        with patch(
            "backend.copilot.sdk.transcript._run_compression",
            side_effect=mock_compression,
        ):
            await compact_transcript(transcript, model="test-model")

        # Check that the messages sent to compression don't contain
        # thinking content from older assistant messages
        for msg in captured_messages:
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                assert (
                    "I should list the files." not in content
                ), "Old thinking block content leaked into compression input"
                assert (
                    "Good, I see two Python files." not in content
                ), "Old thinking block content leaked into compression input"

    @pytest.mark.asyncio
    async def test_trailing_user_message_after_last_assistant(self, mock_chat_config):
        """When the last entry is a user message, the last *assistant*
        message's thinking blocks should still be preserved."""
        transcript = build_structured_transcript(
            [
                ("user", "Hello"),
                (
                    "assistant",
                    [
                        THINKING_BLOCK,
                        {"type": "text", "text": "Hi there"},
                    ],
                ),
                ("user", "Follow-up question"),
            ]
        )

        compacted_msgs = [
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Follow-up question"},
        ]
        mock_result = type(
            "CompressResult",
            (),
            {
                "was_compacted": True,
                "messages": compacted_msgs,
                "original_token_count": 400,
                "token_count": 100,
                "messages_summarized": 1,
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

        last_content = _last_assistant_content(result)
        assert last_content is not None
        assert isinstance(last_content, list)
        block_types = [b["type"] for b in last_content]
        assert (
            "thinking" in block_types
        ), "thinking block lost from last assistant despite trailing user msg"

    @pytest.mark.asyncio
    async def test_single_assistant_with_thinking_preserved(self, mock_chat_config):
        """When there's only one assistant message (which is also the last),
        its thinking blocks must be preserved."""
        transcript = build_structured_transcript(
            [
                ("user", "Hello"),
                (
                    "assistant",
                    [
                        THINKING_BLOCK,
                        {"type": "text", "text": "World"},
                    ],
                ),
            ]
        )

        compacted_msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        mock_result = type(
            "CompressResult",
            (),
            {
                "was_compacted": True,
                "messages": compacted_msgs,
                "original_token_count": 200,
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

        assert result is not None

        last_content = _last_assistant_content(result)
        assert last_content is not None
        assert isinstance(last_content, list)
        block_types = [b["type"] for b in last_content]
        assert "thinking" in block_types

    @pytest.mark.asyncio
    async def test_no_thinking_blocks_still_works(self, mock_chat_config):
        """Compaction should still work normally when there are no thinking
        blocks in the transcript."""
        transcript = build_structured_transcript(
            [
                ("user", "Hello"),
                ("assistant", [{"type": "text", "text": "Hi"}]),
                ("user", "More"),
                ("assistant", [{"type": "text", "text": "Details"}]),
            ]
        )

        compacted_msgs = [
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "Summary"},
        ]
        mock_result = type(
            "CompressResult",
            (),
            {
                "was_compacted": True,
                "messages": compacted_msgs,
                "original_token_count": 200,
                "token_count": 50,
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
        # Verify last assistant content is preserved even without thinking blocks
        last_content = _last_assistant_content(result)
        assert last_content is not None
        assert last_content == [{"type": "text", "text": "Details"}]


# ---------------------------------------------------------------------------
# _transcript_to_messages — thinking block handling
# ---------------------------------------------------------------------------


class TestTranscriptToMessagesThinking:
    def test_thinking_blocks_excluded_from_flattened_content(self):
        """When _transcript_to_messages flattens content, thinking block
        text should not leak into the message content string."""
        transcript = build_structured_transcript(
            [
                ("user", "Hello"),
                (
                    "assistant",
                    [
                        {
                            "type": "thinking",
                            "thinking": "SECRET_THOUGHT",
                            "signature": "sig",
                        },
                        {"type": "text", "text": "Visible response"},
                    ],
                ),
            ]
        )
        messages = _transcript_to_messages(transcript)
        assistant_msg = [m for m in messages if m["role"] == "assistant"][0]
        assert "SECRET_THOUGHT" not in assistant_msg["content"]
        assert "Visible response" in assistant_msg["content"]


# ---------------------------------------------------------------------------
# response_adapter — ThinkingBlock handling
# ---------------------------------------------------------------------------


class TestResponseAdapterThinkingBlock:
    def test_thinking_block_does_not_crash(self):
        """ThinkingBlock in AssistantMessage should not cause an error."""
        adapter = SDKResponseAdapter(message_id="msg-1", session_id="sess-1")
        msg = AssistantMessage(
            content=[
                ThinkingBlock(
                    thinking="Let me think about this...",
                    signature="sig_test_123",
                ),
                TextBlock(text="Here is my response."),
            ],
            model="claude-test",
        )
        results = adapter.convert_message(msg)
        # Should produce stream events for text only, no crash
        types = [type(r) for r in results]
        assert StreamStartStep in types
        assert StreamTextStart in types or StreamTextDelta in types

    def test_thinking_block_does_not_emit_stream_events(self):
        """ThinkingBlock should NOT produce any StreamTextDelta events
        containing thinking content."""
        adapter = SDKResponseAdapter(message_id="msg-1", session_id="sess-1")
        msg = AssistantMessage(
            content=[
                ThinkingBlock(
                    thinking="My secret thoughts",
                    signature="sig_test_456",
                ),
                TextBlock(text="Public response"),
            ],
            model="claude-test",
        )
        results = adapter.convert_message(msg)
        text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
        for delta in text_deltas:
            assert "secret thoughts" not in (delta.delta or "")


# ---------------------------------------------------------------------------
# _format_sdk_content_blocks — redacted_thinking handling
# ---------------------------------------------------------------------------


class TestFormatSdkContentBlocks:
    def test_thinking_block_preserved(self):
        """ThinkingBlock should be serialized with type, thinking, and signature."""
        blocks = [
            ThinkingBlock(thinking="My thoughts", signature="sig123"),
            TextBlock(text="Response"),
        ]
        result = _format_sdk_content_blocks(blocks)
        assert len(result) == 2
        assert result[0] == {
            "type": "thinking",
            "thinking": "My thoughts",
            "signature": "sig123",
        }
        assert result[1] == {"type": "text", "text": "Response"}
