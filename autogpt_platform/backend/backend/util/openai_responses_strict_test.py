"""Explicit function schema policies must survive API format conversion."""

from copy import deepcopy

import pytest

from backend.util.openai_responses import convert_tools_to_responses_format


@pytest.mark.parametrize("strict", [False, True])
def test_preserves_explicit_function_strictness(strict: bool):
    """Preserve explicit strictness without changing the source tool schema."""
    function = {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": strict,
    }
    tools = [{"type": "function", "function": function}]
    original = deepcopy(tools)

    assert convert_tools_to_responses_format(tools) == [
        {"type": "function", **function}
    ]
    assert tools == original


@pytest.mark.parametrize("fields", [{}, {"strict": None}])
def test_leaves_unspecified_strictness_to_provider(fields: dict):
    """Omit missing or null strictness so the provider keeps its default."""
    tools = [{"type": "function", "function": {"name": "simple", **fields}}]

    assert convert_tools_to_responses_format(tools) == [
        {"type": "function", "name": "simple"}
    ]
