"""Tests for OpenAI Responses API helpers."""

from backend.util.openai_responses import (
    CHAT_COMPLETIONS_ONLY_MODELS,
    convert_tools_to_responses_format,
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


class TestChatCompletionsOnlyModels:
    """Tests for the CHAT_COMPLETIONS_ONLY_MODELS constant."""

    def test_is_frozenset(self):
        """CHAT_COMPLETIONS_ONLY_MODELS should be a frozenset (immutable)."""
        assert isinstance(CHAT_COMPLETIONS_ONLY_MODELS, frozenset)

    def test_contains_expected_models(self):
        """Should contain the legacy gpt-3.5-turbo models."""
        expected = {"gpt-3.5-turbo", "gpt-3.5-turbo-0125"}
        assert CHAT_COMPLETIONS_ONLY_MODELS == expected
