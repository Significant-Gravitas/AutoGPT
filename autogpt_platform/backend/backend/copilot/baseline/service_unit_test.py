"""Unit tests for baseline service pure-logic helpers.

These tests cover ``_baseline_conversation_updater`` and ``_BaselineStreamState``
without requiring API keys, database connections, or network access.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionToolParam

from backend.copilot.baseline.service import (
    _baseline_conversation_updater,
    _BaselineStreamState,
    _compress_session_messages,
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


class TestBaselineCostExtraction:
    """Tests for x-total-cost header extraction in _baseline_llm_caller."""

    @pytest.mark.asyncio
    async def test_cost_usd_extracted_from_response_header(self):
        """state.cost_usd is set from x-total-cost header when present."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="gpt-4o-mini")

        # Build a mock raw httpx response with the cost header
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-total-cost": "0.0123"}

        # Build a mock async streaming response that yields no chunks but has
        # a _response attribute pointing to the mock httpx response
        mock_stream_response = MagicMock()
        mock_stream_response._response = mock_raw_response

        async def empty_aiter():
            return
            yield  # make it an async generator

        mock_stream_response.__aiter__ = lambda self: empty_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_stream_response
        )

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.cost_usd == pytest.approx(0.0123)

    @pytest.mark.asyncio
    async def test_cost_usd_accumulates_across_calls(self):
        """cost_usd accumulates when _baseline_llm_caller is called multiple times."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="gpt-4o-mini")

        def make_stream_mock(cost: str) -> MagicMock:
            mock_raw = MagicMock()
            mock_raw.headers = {"x-total-cost": cost}
            mock_stream = MagicMock()
            mock_stream._response = mock_raw

            async def empty_aiter():
                return
                yield

            mock_stream.__aiter__ = lambda self: empty_aiter()
            return mock_stream

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[make_stream_mock("0.01"), make_stream_mock("0.02")]
        )

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "first"}],
                tools=[],
                state=state,
            )
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "second"}],
                tools=[],
                state=state,
            )

        assert state.cost_usd == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_no_cost_when_header_absent(self):
        """state.cost_usd remains None when response has no x-total-cost header."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="gpt-4o-mini")

        mock_raw = MagicMock()
        mock_raw.headers = {}
        mock_stream = MagicMock()
        mock_stream._response = mock_raw

        async def empty_aiter():
            return
            yield

        mock_stream.__aiter__ = lambda self: empty_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.cost_usd is None

    @pytest.mark.asyncio
    async def test_cost_extracted_even_when_stream_raises(self):
        """cost_usd is captured in the finally block even when streaming fails."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="gpt-4o-mini")

        mock_raw = MagicMock()
        mock_raw.headers = {"x-total-cost": "0.005"}
        mock_stream = MagicMock()
        mock_stream._response = mock_raw

        async def failing_aiter():
            raise RuntimeError("stream error")
            yield  # make it an async generator

        mock_stream.__aiter__ = lambda self: failing_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with (
            patch(
                "backend.copilot.baseline.service._get_openai_client",
                return_value=mock_client,
            ),
            pytest.raises(RuntimeError, match="stream error"),
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.cost_usd == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_no_cost_when_api_call_raises_before_stream(self):
        """finally block is safe when response is None (API call failed before yielding)."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="gpt-4o-mini")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with (
            patch(
                "backend.copilot.baseline.service._get_openai_client",
                return_value=mock_client,
            ),
            pytest.raises(RuntimeError, match="connection refused"),
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        # response was never assigned so cost extraction must not raise
        assert state.cost_usd is None

    @pytest.mark.asyncio
    async def test_no_cost_when_header_missing(self):
        """cost_usd remains None when x-total-cost is absent."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="anthropic/claude-sonnet-4")

        mock_raw = MagicMock()
        mock_raw.headers = {}  # no x-total-cost
        mock_stream = MagicMock()
        mock_stream._response = mock_raw

        mock_chunk = MagicMock()
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.prompt_tokens = 1000
        mock_chunk.usage.completion_tokens = 500
        mock_chunk.usage.prompt_tokens_details = None
        mock_chunk.choices = []

        async def chunk_aiter():
            yield mock_chunk

        mock_stream.__aiter__ = lambda self: chunk_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.cost_usd is None

    @pytest.mark.asyncio
    async def test_cache_tokens_extracted_from_usage_details(self):
        """cache tokens are extracted from prompt_tokens_details.cached_tokens."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="openai/gpt-4o")

        mock_raw = MagicMock()
        mock_raw.headers = {"x-total-cost": "0.01"}
        mock_stream = MagicMock()
        mock_stream._response = mock_raw

        # Create a chunk with prompt_tokens_details
        mock_ptd = MagicMock()
        mock_ptd.cached_tokens = 800

        mock_chunk = MagicMock()
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.prompt_tokens = 1000
        mock_chunk.usage.completion_tokens = 200
        mock_chunk.usage.prompt_tokens_details = mock_ptd
        mock_chunk.choices = []

        async def chunk_aiter():
            yield mock_chunk

        mock_stream.__aiter__ = lambda self: chunk_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.turn_cache_read_tokens == 800
        assert state.turn_prompt_tokens == 1000

    @pytest.mark.asyncio
    async def test_cache_creation_tokens_extracted_from_usage_details(self):
        """cache_creation_tokens are extracted from prompt_tokens_details."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="openai/gpt-4o")

        mock_raw = MagicMock()
        mock_raw.headers = {"x-total-cost": "0.01"}
        mock_stream = MagicMock()
        mock_stream._response = mock_raw

        mock_ptd = MagicMock()
        mock_ptd.cached_tokens = 0
        mock_ptd.cache_creation_input_tokens = 500

        mock_chunk = MagicMock()
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.prompt_tokens = 1000
        mock_chunk.usage.completion_tokens = 200
        mock_chunk.usage.prompt_tokens_details = mock_ptd
        mock_chunk.choices = []

        async def chunk_aiter():
            yield mock_chunk

        mock_stream.__aiter__ = lambda self: chunk_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.turn_cache_creation_tokens == 500

    @pytest.mark.asyncio
    async def test_token_accumulators_track_across_multiple_calls(self):
        """Token accumulators grow correctly across multiple _baseline_llm_caller calls."""
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="anthropic/claude-sonnet-4")

        def make_stream(prompt_tokens: int, completion_tokens: int):
            mock_raw = MagicMock()
            mock_raw.headers = {}  # no x-total-cost
            mock_stream = MagicMock()
            mock_stream._response = mock_raw

            mock_chunk = MagicMock()
            mock_chunk.usage = MagicMock()
            mock_chunk.usage.prompt_tokens = prompt_tokens
            mock_chunk.usage.completion_tokens = completion_tokens
            mock_chunk.usage.prompt_tokens_details = None
            mock_chunk.choices = []

            async def chunk_aiter():
                yield mock_chunk

            mock_stream.__aiter__ = lambda self: chunk_aiter()
            return mock_stream

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                make_stream(1000, 200),
                make_stream(1100, 300),
            ]
        )

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "follow up"}],
                tools=[],
                state=state,
            )

        # No x-total-cost header and empty pricing table -- cost_usd remains None
        assert state.cost_usd is None
        # Accumulators hold all tokens across both turns
        assert state.turn_prompt_tokens == 2100
        assert state.turn_completion_tokens == 500

    @pytest.mark.asyncio
    async def test_cost_usd_remains_none_when_header_missing(self):
        """cost_usd stays None when x-total-cost header is absent.

        Token counts are still tracked; persist_and_record_usage handles
        the None cost by falling back to tracking_type='tokens'.
        """
        from backend.copilot.baseline.service import (
            _baseline_llm_caller,
            _BaselineStreamState,
        )

        state = _BaselineStreamState(model="anthropic/claude-sonnet-4")

        mock_raw = MagicMock()
        mock_raw.headers = {}  # no x-total-cost
        mock_stream = MagicMock()
        mock_stream._response = mock_raw

        mock_chunk = MagicMock()
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.prompt_tokens = 1000
        mock_chunk.usage.completion_tokens = 500
        mock_chunk.usage.prompt_tokens_details = None
        mock_chunk.choices = []

        async def chunk_aiter():
            yield mock_chunk

        mock_stream.__aiter__ = lambda self: chunk_aiter()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch(
            "backend.copilot.baseline.service._get_openai_client",
            return_value=mock_client,
        ):
            await _baseline_llm_caller(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                state=state,
            )

        assert state.cost_usd is None
        assert state.turn_prompt_tokens == 1000
        assert state.turn_completion_tokens == 500


class TestMidLoopPendingFlushOrdering:
    """Regression test for the mid-loop pending drain ordering invariant.

    ``_baseline_conversation_updater`` records assistant+tool entries from
    each tool-call round into ``state.session_messages``; the finally block
    of ``stream_chat_completion_baseline`` batch-flushes them into
    ``session.messages`` at the end of the turn.

    The mid-loop pending drain appends pending user messages directly to
    ``session.messages``.  Without flushing ``state.session_messages`` first,
    the pending user message lands BEFORE the preceding round's assistant+
    tool entries in the final persisted ``session.messages`` — which
    produces a malformed tool-call/tool-result ordering on the next turn's
    replay.

    This test documents the invariant by replaying the production flush
    sequence against an in-memory state.
    """

    def test_flush_then_append_preserves_chronological_order(self):
        """Mid-loop drain must flush state.session_messages before appending
        the pending user message, so the final order matches the
        chronological execution order.
        """
        # Initial state: user turn already appended by maybe_append_user_message
        session_messages: list[ChatMessage] = [
            ChatMessage(role="user", content="original user turn"),
        ]
        state = _BaselineStreamState()

        # Round 1 completes: conversation_updater buffers assistant+tool
        # entries into state.session_messages (but does NOT write to
        # session.messages yet).
        builder = TranscriptBuilder()
        builder.append_user("original user turn")
        response = LLMLoopResponse(
            response_text="calling search",
            tool_calls=[LLMToolCall(id="tc_1", name="search", arguments="{}")],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        tool_results = [
            ToolCallResult(
                tool_call_id="tc_1", tool_name="search", content="search output"
            ),
        ]
        openai_messages: list = []
        _baseline_conversation_updater(
            openai_messages,
            response,
            tool_results=tool_results,
            transcript_builder=builder,
            state=state,
            model="test-model",
        )
        # state.session_messages should now hold the round-1 assistant + tool
        assert len(state.session_messages) == 2
        assert state.session_messages[0].role == "assistant"
        assert state.session_messages[1].role == "tool"

        # --- Mid-loop pending drain (production code pattern) ---
        # Flush first, THEN append pending.  This is the ordering fix.
        for _buffered in state.session_messages:
            session_messages.append(_buffered)
        state.session_messages.clear()
        session_messages.append(
            ChatMessage(role="user", content="pending mid-loop message")
        )

        # Round 2 completes: new assistant+tool entries buffer again.
        response2 = LLMLoopResponse(
            response_text="another call",
            tool_calls=[LLMToolCall(id="tc_2", name="calc", arguments="{}")],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        tool_results2 = [
            ToolCallResult(
                tool_call_id="tc_2", tool_name="calc", content="calc output"
            ),
        ]
        _baseline_conversation_updater(
            openai_messages,
            response2,
            tool_results=tool_results2,
            transcript_builder=builder,
            state=state,
            model="test-model",
        )

        # --- Finally-block flush (end of turn) ---
        for msg in state.session_messages:
            session_messages.append(msg)

        # Assert chronological order: original user, round-1 assistant,
        # round-1 tool, pending user, round-2 assistant, round-2 tool.
        assert [m.role for m in session_messages] == [
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
        ]
        assert session_messages[0].content == "original user turn"
        assert session_messages[3].content == "pending mid-loop message"
        # The assistant message carrying tool_call tc_1 must be immediately
        # followed by its tool result — no user message interposed.
        assert session_messages[1].role == "assistant"
        assert session_messages[1].tool_calls is not None
        assert session_messages[1].tool_calls[0]["id"] == "tc_1"
        assert session_messages[2].role == "tool"
        assert session_messages[2].tool_call_id == "tc_1"
        # Same invariant for the round after the pending user.
        assert session_messages[4].tool_calls is not None
        assert session_messages[4].tool_calls[0]["id"] == "tc_2"
        assert session_messages[5].tool_call_id == "tc_2"

    def test_flushed_assistant_text_len_prevents_duplicate_final_text(self):
        """After mid-loop drain clears state.session_messages, the finally
        block must not re-append assistant text from rounds already flushed.

        ``state.assistant_text`` accumulates ALL rounds' text, but
        ``state.session_messages`` only holds entries from rounds AFTER the
        last mid-loop flush.  Without ``_flushed_assistant_text_len``, the
        ``finally`` block's ``startswith(recorded)`` check fails because
        ``recorded`` only covers post-flush rounds, and the full
        ``assistant_text`` is appended — duplicating pre-flush rounds.
        """
        state = _BaselineStreamState()
        session_messages: list[ChatMessage] = [
            ChatMessage(role="user", content="user turn"),
        ]

        # Simulate round 1 text accumulation (as _bound_llm_caller does)
        state.assistant_text += "calling search"

        # Round 1 conversation_updater buffers structured entries
        builder = TranscriptBuilder()
        builder.append_user("user turn")
        response1 = LLMLoopResponse(
            response_text="calling search",
            tool_calls=[LLMToolCall(id="tc_1", name="search", arguments="{}")],
            raw_response=None,
            prompt_tokens=0,
            completion_tokens=0,
        )
        _baseline_conversation_updater(
            [],
            response1,
            tool_results=[
                ToolCallResult(
                    tool_call_id="tc_1", tool_name="search", content="result"
                )
            ],
            transcript_builder=builder,
            state=state,
            model="test-model",
        )

        # Mid-loop drain: flush + clear + record flushed text length
        for _buffered in state.session_messages:
            session_messages.append(_buffered)
        state.session_messages.clear()
        state._flushed_assistant_text_len = len(state.assistant_text)
        session_messages.append(ChatMessage(role="user", content="pending message"))

        # Simulate round 2 text accumulation
        state.assistant_text += "final answer"

        # Round 2: natural finish (no tool calls → no session_messages entry)

        # --- Finally block logic (production code) ---
        for msg in state.session_messages:
            session_messages.append(msg)

        final_text = state.assistant_text[state._flushed_assistant_text_len :]
        if state.session_messages:
            recorded = "".join(
                m.content or "" for m in state.session_messages if m.role == "assistant"
            )
            if final_text.startswith(recorded):
                final_text = final_text[len(recorded) :]
        if final_text.strip():
            session_messages.append(ChatMessage(role="assistant", content=final_text))

        # The final assistant message should only contain round-2 text,
        # not the round-1 text that was already flushed mid-loop.
        assistant_msgs = [m for m in session_messages if m.role == "assistant"]
        # Round-1 structured assistant (from mid-loop flush)
        assert assistant_msgs[0].content == "calling search"
        assert assistant_msgs[0].tool_calls is not None
        # Round-2 final text (from finally block)
        assert assistant_msgs[1].content == "final answer"
        assert assistant_msgs[1].tool_calls is None
        # Crucially: only 2 assistant messages, not 3 (no duplicate)
        assert len(assistant_msgs) == 2
