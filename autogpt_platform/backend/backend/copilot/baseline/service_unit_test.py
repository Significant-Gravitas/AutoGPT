"""Unit tests for baseline service pure-logic helpers.

These tests cover ``_baseline_conversation_updater`` and ``_BaselineStreamState``
without requiring API keys, database connections, or network access.
"""

from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletionToolParam

from backend.copilot.baseline.service import (
    _baseline_conversation_updater,
    _BaselineStreamState,
    _compress_session_messages,
    _ThinkingStripper,
)
from backend.copilot.model import ChatMessage
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util.prompt import CompressResult
from backend.util.tool_call_loop import LLMLoopResponse, LLMToolCall, ToolCallResult


class TestBaselineStreamState:
    def test_defaults(self):
        state = _BaselineStreamState()
        assert state.pending_events == []
        assert state.assistant_text == ""
        assert state.text_started is False
        assert state.turn_prompt_tokens == 0
        assert state.turn_completion_tokens == 0
        assert state.text_block_id  # Should be a UUID string

    def test_mutable_fields(self):
        state = _BaselineStreamState()
        state.assistant_text = "hello"
        state.turn_prompt_tokens = 100
        state.turn_completion_tokens = 50
        assert state.assistant_text == "hello"
        assert state.turn_prompt_tokens == 100
        assert state.turn_completion_tokens == 50


class TestBaselineConversationUpdater:
    """Tests for _baseline_conversation_updater which updates the OpenAI
    message list and transcript builder after each LLM call."""

    def _make_transcript_builder(self) -> TranscriptBuilder:
        builder = TranscriptBuilder()
        builder.append_user("test question")
        return builder

    def test_text_only_response(self):
        """When the LLM returns text without tool calls, the updater appends
        a single assistant message and records it in the transcript."""
        messages: list = []
        builder = self._make_transcript_builder()
        response = LLMLoopResponse(
            response_text="Hello, world!",
            tool_calls=[],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )

        _baseline_conversation_updater(
            messages,
            response,
            tool_results=None,
            transcript_builder=builder,
            model="test-model",
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hello, world!"
        # Transcript should have user + assistant
        assert builder.entry_count == 2
        assert builder.last_entry_type == "assistant"

    def test_tool_calls_response(self):
        """When the LLM returns tool calls, the updater appends the assistant
        message with tool_calls and tool result messages."""
        messages: list = []
        builder = self._make_transcript_builder()
        response = LLMLoopResponse(
            response_text="Let me search...",
            tool_calls=[
                LLMToolCall(
                    id="tc_1",
                    name="search",
                    arguments='{"query": "test"}',
                ),
            ],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        tool_results = [
            ToolCallResult(
                tool_call_id="tc_1",
                tool_name="search",
                content="Found result",
            ),
        ]

        _baseline_conversation_updater(
            messages,
            response,
            tool_results=tool_results,
            transcript_builder=builder,
            model="test-model",
        )

        # Messages: assistant (with tool_calls) + tool result
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Let me search..."
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["tool_calls"][0]["id"] == "tc_1"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "tc_1"
        assert messages[1]["content"] == "Found result"

        # Transcript: user + assistant(tool_use) + user(tool_result)
        assert builder.entry_count == 3

    def test_tool_calls_without_text(self):
        """Tool calls without accompanying text should still work."""
        messages: list = []
        builder = self._make_transcript_builder()
        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[
                LLMToolCall(id="tc_1", name="run", arguments="{}"),
            ],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        tool_results = [
            ToolCallResult(tool_call_id="tc_1", tool_name="run", content="done"),
        ]

        _baseline_conversation_updater(
            messages,
            response,
            tool_results=tool_results,
            transcript_builder=builder,
            model="test-model",
        )

        assert len(messages) == 2
        assert "content" not in messages[0]  # No text content
        assert messages[0]["tool_calls"][0]["function"]["name"] == "run"

    def test_no_text_no_tools(self):
        """When the response has no text and no tool calls, nothing is appended."""
        messages: list = []
        builder = self._make_transcript_builder()
        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )

        _baseline_conversation_updater(
            messages,
            response,
            tool_results=None,
            transcript_builder=builder,
            model="test-model",
        )

        assert len(messages) == 0
        # Only the user entry from setup
        assert builder.entry_count == 1

    def test_multiple_tool_calls(self):
        """Multiple tool calls in a single response are all recorded."""
        messages: list = []
        builder = self._make_transcript_builder()
        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[
                LLMToolCall(id="tc_1", name="tool_a", arguments="{}"),
                LLMToolCall(id="tc_2", name="tool_b", arguments='{"x": 1}'),
            ],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        tool_results = [
            ToolCallResult(tool_call_id="tc_1", tool_name="tool_a", content="result_a"),
            ToolCallResult(tool_call_id="tc_2", tool_name="tool_b", content="result_b"),
        ]

        _baseline_conversation_updater(
            messages,
            response,
            tool_results=tool_results,
            transcript_builder=builder,
            model="test-model",
        )

        # 1 assistant + 2 tool results
        assert len(messages) == 3
        assert len(messages[0]["tool_calls"]) == 2
        assert messages[1]["tool_call_id"] == "tc_1"
        assert messages[2]["tool_call_id"] == "tc_2"

    def test_invalid_tool_arguments_handled(self):
        """Tool call with invalid JSON arguments: the arguments field is
        stored as-is in the message, and orjson failure falls back to {}
        in the transcript content_blocks."""
        messages: list = []
        builder = self._make_transcript_builder()
        response = LLMLoopResponse(
            response_text=None,
            tool_calls=[
                LLMToolCall(id="tc_1", name="tool_x", arguments="not-json"),
            ],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        tool_results = [
            ToolCallResult(tool_call_id="tc_1", tool_name="tool_x", content="ok"),
        ]

        _baseline_conversation_updater(
            messages,
            response,
            tool_results=tool_results,
            transcript_builder=builder,
            model="test-model",
        )

        # Should not raise — invalid JSON falls back to {} in transcript
        assert len(messages) == 2
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == "not-json"


class TestCompressSessionMessagesPreservesToolCalls:
    """``_compress_session_messages`` must round-trip tool_calls + tool_call_id.

    Compression serialises ChatMessage to dict for ``compress_context`` and
    reifies the result back to ChatMessage.  A regression that drops
    ``tool_calls`` or ``tool_call_id`` would corrupt the OpenAI message
    list and break downstream tool-execution rounds.
    """

    @pytest.mark.asyncio
    async def test_compressed_output_keeps_tool_calls_and_ids(self):
        # Simulate compression that returns a summary + the most recent
        # assistant(tool_call) + tool(tool_result) intact.
        summary = {"role": "system", "content": "prior turns: user asked X"}
        assistant_with_tc = {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "tc_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"y"}'},
                }
            ],
        }
        tool_result = {
            "role": "tool",
            "tool_call_id": "tc_abc",
            "content": "search result",
        }

        compress_result = CompressResult(
            messages=[summary, assistant_with_tc, tool_result],
            token_count=100,
            was_compacted=True,
            original_token_count=5000,
            messages_summarized=10,
            messages_dropped=0,
        )

        # Input: messages that should be compressed.
        input_messages = [
            ChatMessage(role="user", content="q1"),
            ChatMessage(
                role="assistant",
                content="calling tool",
                tool_calls=[
                    {
                        "id": "tc_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"y"}',
                        },
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="tc_abc",
                content="search result",
            ),
        ]

        with patch(
            "backend.copilot.baseline.service.compress_context",
            new=AsyncMock(return_value=compress_result),
        ):
            compressed = await _compress_session_messages(
                input_messages, model="openrouter/anthropic/claude-opus-4"
            )

        # Summary, assistant(tool_calls), tool(tool_call_id).
        assert len(compressed) == 3
        # Assistant message must keep its tool_calls intact.
        assistant_msg = compressed[1]
        assert assistant_msg.role == "assistant"
        assert assistant_msg.tool_calls is not None
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0]["id"] == "tc_abc"
        assert assistant_msg.tool_calls[0]["function"]["name"] == "search"
        # Tool-role message must keep tool_call_id for OpenAI linkage.
        tool_msg = compressed[2]
        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "tc_abc"
        assert tool_msg.content == "search result"

    @pytest.mark.asyncio
    async def test_uncompressed_passthrough_keeps_fields(self):
        """When compression is a no-op (was_compacted=False), the original
        messages must be returned unchanged — including tool_calls."""
        input_messages = [
            ChatMessage(
                role="assistant",
                content="c",
                tool_calls=[
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(role="tool", tool_call_id="t1", content="ok"),
        ]

        noop_result = CompressResult(
            messages=[],  # ignored when was_compacted=False
            token_count=10,
            was_compacted=False,
        )

        with patch(
            "backend.copilot.baseline.service.compress_context",
            new=AsyncMock(return_value=noop_result),
        ):
            out = await _compress_session_messages(
                input_messages, model="openrouter/anthropic/claude-opus-4"
            )

        assert out is input_messages  # same list returned
        assert out[0].tool_calls is not None
        assert out[0].tool_calls[0]["id"] == "t1"
        assert out[1].tool_call_id == "t1"


# ---- _ThinkingStripper tests ---- #


def test_thinking_stripper_basic_thinking_tag() -> None:
    """<thinking>...</thinking> blocks are fully stripped."""
    s = _ThinkingStripper()
    assert s.process("<thinking>internal reasoning here</thinking>Hello!") == "Hello!"


def test_thinking_stripper_internal_reasoning_tag() -> None:
    """<internal_reasoning>...</internal_reasoning> blocks (Gemini) are stripped."""
    s = _ThinkingStripper()
    assert (
        s.process("<internal_reasoning>step by step</internal_reasoning>Answer")
        == "Answer"
    )


def test_thinking_stripper_split_across_chunks() -> None:
    """Tags split across multiple chunks are handled correctly."""
    s = _ThinkingStripper()
    out = s.process("Hello <thin")
    out += s.process("king>secret</thinking> world")
    assert out == "Hello  world"


def test_thinking_stripper_plain_text_preserved() -> None:
    """Plain text with the word 'thinking' is not stripped."""
    s = _ThinkingStripper()
    assert (
        s.process("I am thinking about this problem")
        == "I am thinking about this problem"
    )


def test_thinking_stripper_multiple_blocks() -> None:
    """Multiple reasoning blocks in one stream are all stripped."""
    s = _ThinkingStripper()
    result = s.process(
        "A<thinking>x</thinking>B<internal_reasoning>y</internal_reasoning>C"
    )
    assert result == "ABC"


def test_thinking_stripper_flush_discards_unclosed() -> None:
    """Unclosed reasoning block is discarded on flush."""
    s = _ThinkingStripper()
    s.process("Start<thinking>never closed")
    flushed = s.flush()
    assert "never closed" not in flushed


def test_thinking_stripper_empty_block() -> None:
    """Empty reasoning blocks are handled gracefully."""
    s = _ThinkingStripper()
    assert s.process("Before<thinking></thinking>After") == "BeforeAfter"


# ---- _filter_tools_by_permissions tests ---- #


def _make_tool(name: str) -> ChatCompletionToolParam:
    """Build a minimal OpenAI ChatCompletionToolParam."""
    return ChatCompletionToolParam(
        type="function",
        function={"name": name, "parameters": {}},
    )


class TestFilterToolsByPermissions:
    """Tests for _filter_tools_by_permissions."""

    @patch(
        "backend.copilot.permissions.all_known_tool_names",
        return_value=frozenset({"run_block", "web_fetch", "bash_exec"}),
    )
    def test_empty_permissions_returns_all(self, _mock_names):
        """Empty permissions (no filtering) returns every tool unchanged."""
        from backend.copilot.baseline.service import _filter_tools_by_permissions
        from backend.copilot.permissions import CopilotPermissions

        tools = [_make_tool("run_block"), _make_tool("web_fetch")]
        perms = CopilotPermissions()
        result = _filter_tools_by_permissions(tools, perms)
        assert result == tools

    @patch(
        "backend.copilot.permissions.all_known_tool_names",
        return_value=frozenset({"run_block", "web_fetch", "bash_exec"}),
    )
    def test_allowlist_keeps_only_matching(self, _mock_names):
        """Explicit allowlist (tools_exclude=False) keeps only listed tools."""
        from backend.copilot.baseline.service import _filter_tools_by_permissions
        from backend.copilot.permissions import CopilotPermissions

        tools = [
            _make_tool("run_block"),
            _make_tool("web_fetch"),
            _make_tool("bash_exec"),
        ]
        perms = CopilotPermissions(tools=["web_fetch"], tools_exclude=False)
        result = _filter_tools_by_permissions(tools, perms)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_fetch"

    @patch(
        "backend.copilot.permissions.all_known_tool_names",
        return_value=frozenset({"run_block", "web_fetch", "bash_exec"}),
    )
    def test_blacklist_excludes_listed(self, _mock_names):
        """Blacklist (tools_exclude=True) removes only the listed tools."""
        from backend.copilot.baseline.service import _filter_tools_by_permissions
        from backend.copilot.permissions import CopilotPermissions

        tools = [
            _make_tool("run_block"),
            _make_tool("web_fetch"),
            _make_tool("bash_exec"),
        ]
        perms = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        result = _filter_tools_by_permissions(tools, perms)
        names = [t["function"]["name"] for t in result]
        assert "bash_exec" not in names
        assert "run_block" in names
        assert "web_fetch" in names
        assert len(result) == 2

    @patch(
        "backend.copilot.permissions.all_known_tool_names",
        return_value=frozenset({"run_block", "web_fetch", "bash_exec"}),
    )
    def test_unknown_tool_name_filtered_out(self, _mock_names):
        """A tool whose name is not in all_known_tool_names is dropped."""
        from backend.copilot.baseline.service import _filter_tools_by_permissions
        from backend.copilot.permissions import CopilotPermissions

        tools = [_make_tool("run_block"), _make_tool("unknown_tool")]
        perms = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        result = _filter_tools_by_permissions(tools, perms)
        names = [t["function"]["name"] for t in result]
        assert "unknown_tool" not in names
        assert names == ["run_block"]


# ---- _prepare_baseline_attachments tests ---- #


class TestPrepareBaselineAttachments:
    """Tests for _prepare_baseline_attachments."""

    @pytest.mark.asyncio
    async def test_empty_file_ids(self):
        """Empty file_ids returns empty hint and blocks."""
        from backend.copilot.baseline.service import _prepare_baseline_attachments

        hint, blocks = await _prepare_baseline_attachments([], "user1", "sess1", "/tmp")
        assert hint == ""
        assert blocks == []

    @pytest.mark.asyncio
    async def test_empty_user_id(self):
        """Empty user_id returns empty hint and blocks."""
        from backend.copilot.baseline.service import _prepare_baseline_attachments

        hint, blocks = await _prepare_baseline_attachments(
            ["file1"], "", "sess1", "/tmp"
        )
        assert hint == ""
        assert blocks == []

    @pytest.mark.asyncio
    async def test_image_file_returns_vision_blocks(self):
        """A PNG image within size limits is returned as a base64 vision block."""
        from backend.copilot.baseline.service import _prepare_baseline_attachments

        fake_info = AsyncMock()
        fake_info.name = "photo.png"
        fake_info.mime_type = "image/png"
        fake_info.size_bytes = 1024

        fake_manager = AsyncMock()
        fake_manager.get_file_info = AsyncMock(return_value=fake_info)
        fake_manager.read_file_by_id = AsyncMock(return_value=b"\x89PNG_FAKE_DATA")

        with patch(
            "backend.copilot.baseline.service.get_workspace_manager",
            new=AsyncMock(return_value=fake_manager),
        ):
            hint, blocks = await _prepare_baseline_attachments(
                ["fid1"], "user1", "sess1", "/tmp/workdir"
            )

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["media_type"] == "image/png"
        assert blocks[0]["source"]["type"] == "base64"
        assert "photo.png" in hint
        assert "embedded as image" in hint

    @pytest.mark.asyncio
    async def test_non_image_file_saved_to_working_dir(self, tmp_path):
        """A non-image file is written to working_dir."""
        from backend.copilot.baseline.service import _prepare_baseline_attachments

        fake_info = AsyncMock()
        fake_info.name = "data.csv"
        fake_info.mime_type = "text/csv"
        fake_info.size_bytes = 42

        fake_manager = AsyncMock()
        fake_manager.get_file_info = AsyncMock(return_value=fake_info)
        fake_manager.read_file_by_id = AsyncMock(return_value=b"col1,col2\na,b")

        with patch(
            "backend.copilot.baseline.service.get_workspace_manager",
            new=AsyncMock(return_value=fake_manager),
        ):
            hint, blocks = await _prepare_baseline_attachments(
                ["fid1"], "user1", "sess1", str(tmp_path)
            )

        assert blocks == []
        assert "data.csv" in hint
        assert "saved to" in hint
        saved = tmp_path / "data.csv"
        assert saved.exists()
        assert saved.read_bytes() == b"col1,col2\na,b"

    @pytest.mark.asyncio
    async def test_file_not_found_skipped(self):
        """When get_file_info returns None the file is silently skipped."""
        from backend.copilot.baseline.service import _prepare_baseline_attachments

        fake_manager = AsyncMock()
        fake_manager.get_file_info = AsyncMock(return_value=None)

        with patch(
            "backend.copilot.baseline.service.get_workspace_manager",
            new=AsyncMock(return_value=fake_manager),
        ):
            hint, blocks = await _prepare_baseline_attachments(
                ["missing_id"], "user1", "sess1", "/tmp"
            )

        assert hint == ""
        assert blocks == []

    @pytest.mark.asyncio
    async def test_workspace_manager_error(self):
        """When get_workspace_manager raises, returns empty results."""
        from backend.copilot.baseline.service import _prepare_baseline_attachments

        with patch(
            "backend.copilot.baseline.service.get_workspace_manager",
            new=AsyncMock(side_effect=RuntimeError("connection failed")),
        ):
            hint, blocks = await _prepare_baseline_attachments(
                ["fid1"], "user1", "sess1", "/tmp"
            )

        assert hint == ""
        assert blocks == []
