"""Tests for the OpenAI → Anthropic tool conversion ``call_provider`` runs."""

from __future__ import annotations

import anthropic
from pydantic import BaseModel

from backend.util.llm.conversions import convert_openai_tool_fmt_to_anthropic
from backend.util.llm.tool_use import pydantic_to_anthropic_tool

_LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_LOOKUP_TOOL = {
    "name": "lookup",
    "description": "Look something up.",
    "input_schema": _LOOKUP_SCHEMA,
}


class _Fact(BaseModel):
    content: str
    confidence: float


class _Output(BaseModel):
    facts: list[_Fact]
    summary: str | None = None


def test_no_tools_leaves_the_field_out():
    assert convert_openai_tool_fmt_to_anthropic(None) is anthropic.NOT_GIVEN
    assert convert_openai_tool_fmt_to_anthropic([]) is anthropic.NOT_GIVEN


def test_openai_function_tool_keeps_its_parameters():
    converted = convert_openai_tool_fmt_to_anthropic(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": _LOOKUP_SCHEMA,
                },
            }
        ]
    )
    assert converted == [_LOOKUP_TOOL]


def test_raw_function_def_keeps_its_parameters():
    converted = convert_openai_tool_fmt_to_anthropic(
        [
            {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": _LOOKUP_SCHEMA,
            }
        ]
    )
    assert converted == [_LOOKUP_TOOL]


def test_anthropic_shaped_tool_keeps_its_input_schema():
    """``pydantic_to_anthropic_tool`` builds ``input_schema``; reading only
    ``parameters`` sent such a tool with an empty schema (the dream batch
    path's forced tool, for one)."""
    tool = pydantic_to_anthropic_tool(
        _Output, tool_name="emit_output", description="Emit the output."
    )
    schema = tool["input_schema"]

    converted = convert_openai_tool_fmt_to_anthropic([tool])

    assert converted == [
        {
            "name": "emit_output",
            "description": "Emit the output.",
            "input_schema": {
                "type": "object",
                "properties": schema["properties"],
                "required": ["facts"],
            },
        }
    ]
    # The nested model survives inside the properties, inlined.
    assert set(schema["properties"]["facts"]["items"]["properties"]) == {
        "content",
        "confidence",
    }
