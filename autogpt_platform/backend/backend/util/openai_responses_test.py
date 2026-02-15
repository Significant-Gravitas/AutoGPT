"""Tests for OpenAI Responses API helpers."""

import pytest

from backend.util.openai_responses import (
    REASONING_MODELS,
    convert_tools_to_responses_format,
    requires_responses_api,
)


class TestRequiresResponsesApi:
    """Tests for the requires_responses_api function."""

    def test_o1_models_require_responses_api(self):
        """O1 family models should require the Responses API."""
        assert requires_responses_api("o1") is True
        assert requires_responses_api("o1-mini") is True
        assert requires_responses_api("o1-preview") is True
        assert requires_responses_api("o1-2024-12-17") is True

    def test_o3_models_require_responses_api(self):
        """O3 family models should require the Responses API."""
        assert requires_responses_api("o3") is True
        assert requires_responses_api("o3-mini") is True
        assert requires_responses_api("o3-2025-04-16") is True
        assert requires_responses_api("o3-mini-2025-01-31") is True

    def test_gpt_models_do_not_require_responses_api(self):
        """GPT models should NOT require the Responses API."""
        assert requires_responses_api("gpt-4o") is False
        assert requires_responses_api("gpt-4o-mini") is False
        assert requires_responses_api("gpt-4-turbo") is False
        assert requires_responses_api("gpt-3.5-turbo") is False
        assert requires_responses_api("gpt-5") is False
        assert requires_responses_api("gpt-5-mini") is False

    def test_other_models_do_not_require_responses_api(self):
        """Other provider models should NOT require the Responses API."""
        assert requires_responses_api("claude-3-opus") is False
        assert requires_responses_api("llama-3.3-70b") is False
        assert requires_responses_api("gemini-pro") is False

    def test_empty_string_does_not_require_responses_api(self):
        """Empty string should not require the Responses API."""
        assert requires_responses_api("") is False

    def test_exact_matching_no_false_positives(self):
        """Should not match models that just start with 'o1' or 'o3'."""
        # These are hypothetical models that start with o1/o3 but aren't
        # actually reasoning models
        assert requires_responses_api("o1-turbo-hypothetical") is False
        assert requires_responses_api("o3-fast-hypothetical") is False
        assert requires_responses_api("o100") is False


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


class TestReasoningModelsSet:
    """Tests for the REASONING_MODELS constant."""

    def test_reasoning_models_is_frozenset(self):
        """REASONING_MODELS should be a frozenset (immutable)."""
        assert isinstance(REASONING_MODELS, frozenset)

    def test_contains_expected_models(self):
        """Should contain all expected reasoning models."""
        expected = {
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-2024-12-17",
            "o3",
            "o3-mini",
            "o3-2025-04-16",
            "o3-mini-2025-01-31",
        }
        assert expected.issubset(REASONING_MODELS)
