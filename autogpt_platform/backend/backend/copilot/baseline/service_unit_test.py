"""Unit tests for baseline service pure-logic helpers.

These tests cover ``_baseline_conversation_updater`` and ``_BaselineStreamState``
without requiring API keys, database connections, or network access.
"""

from backend.copilot.baseline.service import (
    _baseline_conversation_updater,
    _BaselineStreamState,
)
from backend.copilot.transcript_builder import TranscriptBuilder
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
