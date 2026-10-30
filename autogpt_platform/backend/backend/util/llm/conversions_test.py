"""Tests for the OpenAI → Anthropic tool conversion ``call_provider`` runs."""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict

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


class _StrictOutput(BaseModel):
    """Facts, and nothing else."""

    model_config = ConfigDict(extra="forbid")

    facts: list[_Fact]


def _input_schema(tool: dict[str, Any]) -> dict[str, object]:
    converted = convert_openai_tool_fmt_to_anthropic([tool])
    assert isinstance(converted, list)
    return dict(converted[0]["input_schema"])


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


def test_the_whole_schema_reaches_anthropic():
    """A response model with ``extra="forbid"`` keeps its root
    ``additionalProperties: false``, and its docstring the root
    ``description``: the schema is forwarded, not rebuilt."""
    tool = pydantic_to_anthropic_tool(
        _StrictOutput, tool_name="emit_strict", description="Emit it."
    )

    schema = _input_schema(tool)

    assert schema["additionalProperties"] is False
    assert schema["description"] == "Facts, and nothing else."
    assert schema == tool["input_schema"]


def test_openai_parameters_keep_their_defs():
    """Parameters taken straight from ``model_json_schema()`` point their
    ``$ref``s into ``$defs``, so the definitions travel with them."""
    parameters = _Output.model_json_schema()
    assert "$defs" in parameters

    schema = _input_schema(
        {
            "type": "function",
            "function": {"name": "emit_output", "parameters": parameters},
        }
    )

    assert schema["$defs"] == parameters["$defs"]
    assert schema["properties"] == parameters["properties"]


def test_openai_strict_flag_is_not_sent_as_schema():
    """``strict`` inside ``parameters`` is OpenAI's tool flag, not JSON
    Schema; the rest of the schema still goes over."""
    schema = _input_schema(
        {
            "name": "lookup",
            "parameters": {
                **_LOOKUP_SCHEMA,
                "additionalProperties": False,
                "strict": True,
            },
        }
    )

    assert "strict" not in schema
    assert schema["additionalProperties"] is False


def test_empty_parameters_become_an_empty_object_schema():
    assert _input_schema({"name": "ping", "parameters": {}}) == {
        "type": "object",
        "properties": {},
        "required": [],
    }
