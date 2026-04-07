"""Tests for empty-choices guard in extract_openai_tool_calls() and extract_openai_reasoning()."""

from unittest.mock import MagicMock

from backend.blocks.llm import extract_openai_reasoning, extract_openai_tool_calls


class TestExtractOpenaiToolCallsEmptyChoices:
    """extract_openai_tool_calls() must return None when choices is empty."""

    def test_returns_none_for_empty_choices(self):
        response = MagicMock()
        response.choices = []
        assert extract_openai_tool_calls(response) is None

    def test_returns_none_for_none_choices(self):
        response = MagicMock()
        response.choices = None
        assert extract_openai_tool_calls(response) is None

    def test_returns_tool_calls_when_choices_present(self):
        tool = MagicMock()
        tool.id = "call_1"
        tool.type = "function"
        tool.function.name = "my_func"
        tool.function.arguments = '{"a": 1}'

        message = MagicMock()
        message.tool_calls = [tool]

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]

        result = extract_openai_tool_calls(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "my_func"

    def test_returns_none_when_no_tool_calls(self):
        message = MagicMock()
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]

        assert extract_openai_tool_calls(response) is None


class TestExtractOpenaiReasoningEmptyChoices:
    """extract_openai_reasoning() must return None when choices is empty."""

    def test_returns_none_for_empty_choices(self):
        response = MagicMock()
        response.choices = []
        assert extract_openai_reasoning(response) is None

    def test_returns_none_for_none_choices(self):
        response = MagicMock()
        response.choices = None
        assert extract_openai_reasoning(response) is None

    def test_returns_reasoning_from_choice(self):
        choice = MagicMock()
        choice.reasoning = "Step-by-step reasoning"
        choice.message = MagicMock(spec=[])  # no 'reasoning' attr on message

        response = MagicMock(spec=[])  # no 'reasoning' attr on response
        response.choices = [choice]

        result = extract_openai_reasoning(response)
        assert result == "Step-by-step reasoning"

    def test_returns_none_when_no_reasoning(self):
        choice = MagicMock(spec=[])  # no 'reasoning' attr
        choice.message = MagicMock(spec=[])  # no 'reasoning' attr

        response = MagicMock(spec=[])  # no 'reasoning' attr
        response.choices = [choice]

        result = extract_openai_reasoning(response)
        assert result is None
