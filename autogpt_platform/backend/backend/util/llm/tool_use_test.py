"""Tests for ``util.llm.tool_use``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.util.llm.tool_use import (
    _inline_refs,
    force_tool_choice,
    pydantic_to_anthropic_tool,
)


class _Demotion(BaseModel):
    edge_uuid: str
    reason: str
    new_status: Literal["superseded", "contradicted"] = "superseded"


class _Operations(BaseModel):
    writes: list[str] = Field(default_factory=list)
    demotions: list[_Demotion] = Field(default_factory=list)
    summary_for_user: str = ""


class TestPydanticToAnthropicTool:
    def test_returns_name_description_and_input_schema(self):
        tool = pydantic_to_anthropic_tool(
            _Operations,
            tool_name="submit_dream_ops",
            description="Submit the dream pass operations.",
        )
        assert tool["name"] == "submit_dream_ops"
        assert tool["description"] == "Submit the dream pass operations."
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"

    def test_input_schema_includes_required_top_level_properties(self):
        """If we lose `writes`/`demotions`/`summary_for_user` from the
        schema, Anthropic will accept any shape and our parser breaks."""
        tool = pydantic_to_anthropic_tool(_Operations, tool_name="x", description="x")
        props = tool["input_schema"]["properties"]
        assert "writes" in props
        assert "demotions" in props
        assert "summary_for_user" in props

    def test_inlines_nested_model_refs(self):
        """The whole point of the helper: nested model `_Demotion` must
        be inlined into the demotions array's items schema, not left as
        a `$ref` pointing into a removed `$defs` table."""
        tool = pydantic_to_anthropic_tool(_Operations, tool_name="x", description="x")
        demotions_items = tool["input_schema"]["properties"]["demotions"]["items"]
        # After inlining we should see the concrete _Demotion shape, not a $ref
        assert "$ref" not in demotions_items
        assert demotions_items["type"] == "object"
        assert "edge_uuid" in demotions_items["properties"]

    def test_strips_defs_block_after_inlining(self):
        tool = pydantic_to_anthropic_tool(_Operations, tool_name="x", description="x")
        assert "$defs" not in tool["input_schema"]

    def test_strips_title_after_inlining(self):
        """Title pollutes the schema and Anthropic ignores it; drop it."""
        tool = pydantic_to_anthropic_tool(_Operations, tool_name="x", description="x")
        assert "title" not in tool["input_schema"]


class TestForceToolChoice:
    def test_returns_anthropic_forced_choice_shape(self):
        choice = force_tool_choice("submit_dream_ops")
        assert choice["type"] == "tool"
        assert choice["name"] == "submit_dream_ops"

    def test_disables_parallel_tool_use(self):
        """The whole reason we use forced tool_choice is to get exactly
        one tool_use block with no preamble. Parallel calls would
        re-introduce the multi-block output we're trying to eliminate."""
        choice = force_tool_choice("any_name")
        assert choice["disable_parallel_tool_use"] is True


class TestInlineRefs:
    def test_passes_through_schema_without_refs(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert _inline_refs(schema) == schema

    def test_inlines_single_ref(self):
        schema = {
            "type": "object",
            "properties": {"foo": {"$ref": "#/$defs/Foo"}},
            "$defs": {"Foo": {"type": "string", "minLength": 1}},
        }
        result = _inline_refs(schema)
        assert result["properties"]["foo"] == {"type": "string", "minLength": 1}

    def test_inlines_nested_refs_in_arrays(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
            },
            "$defs": {
                "Item": {"type": "object", "properties": {"x": {"type": "integer"}}}
            },
        }
        result = _inline_refs(schema)
        assert result["properties"]["items"]["items"]["type"] == "object"
        assert (
            result["properties"]["items"]["items"]["properties"]["x"]["type"]
            == "integer"
        )

    def test_leaves_unknown_ref_form_alone(self):
        """A ref to an external schema (not #/$defs/...) shouldn't crash;
        we leave it as-is and let Anthropic decide."""
        schema = {"$ref": "http://example.com/schemas/foo"}
        assert _inline_refs(schema) == schema
