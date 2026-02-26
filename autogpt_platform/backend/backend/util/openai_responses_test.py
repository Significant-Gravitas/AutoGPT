"""Tests for OpenAI Responses API helpers."""

from unittest.mock import MagicMock

from backend.util.openai_responses import (
    CHAT_COMPLETIONS_ONLY_MODELS,
    convert_tools_to_responses_format,
    extract_responses_content,
    extract_responses_reasoning,
    extract_responses_tool_calls,
    extract_responses_usage,
    requires_chat_completions,
)


class TestRequiresChatCompletions:
    """Tests for the requires_chat_completions function."""

    def test_gpt35_requires_chat_completions(self):
        """gpt-3.5-turbo models should require Chat Completions API."""
        assert requires_chat_completions("gpt-3.5-turbo") is True
        assert requires_chat_completions("gpt-3.5-turbo-0125") is True

    def test_modern_gpt_models_do_not_require_chat_completions(self):
        """Modern GPT models should NOT require Chat Completions (use Responses API)."""
        assert requires_chat_completions("gpt-4o") is False
        assert requires_chat_completions("gpt-4o-mini") is False
        assert requires_chat_completions("gpt-4-turbo") is False
        assert requires_chat_completions("gpt-4.1-2025-04-14") is False
        assert requires_chat_completions("gpt-5-2025-08-07") is False
        assert requires_chat_completions("gpt-5-mini-2025-08-07") is False

    def test_reasoning_models_do_not_require_chat_completions(self):
        """Reasoning models should NOT require Chat Completions (use Responses API)."""
        assert requires_chat_completions("o1") is False
        assert requires_chat_completions("o1-mini") is False
        assert requires_chat_completions("o3") is False
        assert requires_chat_completions("o3-mini") is False

    def test_other_models_do_not_require_chat_completions(self):
        """Other provider models should NOT require Chat Completions."""
        assert requires_chat_completions("claude-3-opus") is False
        assert requires_chat_completions("llama-3.3-70b") is False
        assert requires_chat_completions("gemini-pro") is False

    def test_empty_string_does_not_require_chat_completions(self):
        """Empty string should not require Chat Completions."""
        assert requires_chat_completions("") is False


class TestConvertToolsToResponsesFormat:
    """Tests for the convert_tools_to_responses_format function."""

    def test_empty_tools_returns_empty_list(self):
        """Empty or None tools should return empty list."""
        assert convert_tools_to_responses_format(None) == []
        assert convert_tools_to_responses_format([]) == []

    def test_converts_function_tool_format(self):
        """Should convert Chat Completions function format to Responses format."""
        chat_completions_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = convert_tools_to_responses_format(chat_completions_tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get the weather in a location"
        assert result[0]["parameters"] == {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        }
        # Should not have nested "function" key
        assert "function" not in result[0]

    def test_handles_multiple_tools(self):
        """Should handle multiple tools."""
        chat_completions_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_1",
                    "description": "First tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_2",
                    "description": "Second tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = convert_tools_to_responses_format(chat_completions_tools)

        assert len(result) == 2
        assert result[0]["name"] == "tool_1"
        assert result[1]["name"] == "tool_2"

    def test_passes_through_non_function_tools(self):
        """Non-function tools should be passed through as-is."""
        tools = [{"type": "web_search", "config": {"enabled": True}}]

        result = convert_tools_to_responses_format(tools)

        assert result == tools

    def test_omits_none_description_and_parameters(self):
        """Should omit description and parameters when they are None."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                },
            }
        ]

        result = convert_tools_to_responses_format(tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "simple_tool"
        assert "description" not in result[0]
        assert "parameters" not in result[0]


class TestExtractResponsesToolCalls:
    """Tests for the extract_responses_tool_calls function."""

    def test_extracts_function_call_items(self):
        """Should extract function_call items from response output."""
        item = MagicMock()
        item.type = "function_call"
        item.call_id = "call_123"
        item.name = "get_weather"
        item.arguments = '{"location": "NYC"}'

        response = MagicMock()
        response.output = [item]

        result = extract_responses_tool_calls(response)

        assert result == [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
            }
        ]

    def test_returns_none_when_no_tool_calls(self):
        """Should return None when no function_call items exist."""
        message_item = MagicMock()
        message_item.type = "message"

        response = MagicMock()
        response.output = [message_item]

        assert extract_responses_tool_calls(response) is None

    def test_returns_none_for_empty_output(self):
        """Should return None when output is empty."""
        response = MagicMock()
        response.output = []

        assert extract_responses_tool_calls(response) is None

    def test_extracts_multiple_tool_calls(self):
        """Should extract multiple function_call items."""
        item1 = MagicMock()
        item1.type = "function_call"
        item1.call_id = "call_1"
        item1.name = "tool_a"
        item1.arguments = "{}"

        item2 = MagicMock()
        item2.type = "function_call"
        item2.call_id = "call_2"
        item2.name = "tool_b"
        item2.arguments = '{"x": 1}'

        response = MagicMock()
        response.output = [item1, item2]

        result = extract_responses_tool_calls(response)

        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"


class TestExtractResponsesUsage:
    """Tests for the extract_responses_usage function."""

    def test_extracts_token_counts(self):
        """Should extract input_tokens and output_tokens."""
        response = MagicMock()
        response.usage.input_tokens = 42
        response.usage.output_tokens = 17

        result = extract_responses_usage(response)

        assert result == (42, 17)

    def test_returns_zeros_when_usage_is_none(self):
        """Should return (0, 0) when usage is None."""
        response = MagicMock()
        response.usage = None

        result = extract_responses_usage(response)

        assert result == (0, 0)


class TestExtractResponsesContent:
    """Tests for the extract_responses_content function."""

    def test_extracts_from_output_text(self):
        """Should use output_text property when available."""
        response = MagicMock()
        response.output_text = "Hello world"

        assert extract_responses_content(response) == "Hello world"

    def test_returns_empty_string_when_output_text_is_none(self):
        """Should return empty string when output_text is None."""
        response = MagicMock()
        response.output_text = None
        response.output = []

        assert extract_responses_content(response) == ""

    def test_fallback_to_output_items(self):
        """Should fall back to extracting from output items."""
        text_content = MagicMock()
        text_content.type = "output_text"
        text_content.text = "Fallback content"

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_content]

        response = MagicMock(spec=[])  # no output_text attribute
        response.output = [message_item]

        assert extract_responses_content(response) == "Fallback content"

    def test_returns_empty_string_for_empty_output(self):
        """Should return empty string when no content found."""
        response = MagicMock(spec=[])  # no output_text attribute
        response.output = []

        assert extract_responses_content(response) == ""


class TestExtractResponsesReasoning:
    """Tests for the extract_responses_reasoning function."""

    def test_extracts_reasoning_summary(self):
        """Should extract reasoning text from summary items."""
        summary_item = MagicMock()
        summary_item.text = "Step 1: Think about it"

        reasoning_item = MagicMock()
        reasoning_item.type = "reasoning"
        reasoning_item.summary = [summary_item]

        response = MagicMock()
        response.output = [reasoning_item]

        assert extract_responses_reasoning(response) == "Step 1: Think about it"

    def test_joins_multiple_summary_items(self):
        """Should join multiple summary text items with newlines."""
        s1 = MagicMock()
        s1.text = "First thought"
        s2 = MagicMock()
        s2.text = "Second thought"

        reasoning_item = MagicMock()
        reasoning_item.type = "reasoning"
        reasoning_item.summary = [s1, s2]

        response = MagicMock()
        response.output = [reasoning_item]

        assert extract_responses_reasoning(response) == "First thought\nSecond thought"

    def test_returns_none_when_no_reasoning(self):
        """Should return None when no reasoning items exist."""
        message_item = MagicMock()
        message_item.type = "message"

        response = MagicMock()
        response.output = [message_item]

        assert extract_responses_reasoning(response) is None

    def test_returns_none_for_empty_output(self):
        """Should return None when output is empty."""
        response = MagicMock()
        response.output = []

        assert extract_responses_reasoning(response) is None

    def test_returns_none_when_summary_is_empty(self):
        """Should return None when reasoning item has empty summary."""
        reasoning_item = MagicMock()
        reasoning_item.type = "reasoning"
        reasoning_item.summary = []

        response = MagicMock()
        response.output = [reasoning_item]

        assert extract_responses_reasoning(response) is None


class TestChatCompletionsOnlyModels:
    """Tests for the CHAT_COMPLETIONS_ONLY_MODELS constant."""

    def test_is_frozenset(self):
        """CHAT_COMPLETIONS_ONLY_MODELS should be a frozenset (immutable)."""
        assert isinstance(CHAT_COMPLETIONS_ONLY_MODELS, frozenset)

    def test_contains_expected_models(self):
        """Should contain the legacy gpt-3.5-turbo models."""
        expected = {"gpt-3.5-turbo", "gpt-3.5-turbo-0125"}
        assert CHAT_COMPLETIONS_ONLY_MODELS == expected
