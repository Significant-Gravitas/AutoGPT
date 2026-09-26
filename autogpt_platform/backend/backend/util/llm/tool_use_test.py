"""Tests for ``util.llm.tool_use``."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from backend.util.llm.tool_use import (
    _inline_refs,
    auto_tool_choice,
    force_tool_choice,
    is_forced_tool_choice,
    pydantic_to_anthropic_tool,
    structured_tool_choice,
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


class TestStructuredToolChoice:
    """Opus 5.5 answers a forced ``tool_choice`` with a 400, so its output
    tool goes out under ``auto``; models that accept forcing keep it."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5-5",
            "anthropic/claude-opus-5.5",
            "anthropic/claude-opus-5-5",
            "claude-opus-5-5-20261015",
            "claude-fable-5-1",
        ],
    )
    def test_models_that_reject_forcing_get_auto(self, model: str):
        assert structured_tool_choice(model, "emit_x") == auto_tool_choice()

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-5",
            "anthropic/claude-sonnet-5",
            "claude-opus-5",
            "claude-haiku-4-5-20251001",
        ],
    )
    def test_models_that_accept_forcing_keep_it(self, model: str):
        assert structured_tool_choice(model, "emit_x") == force_tool_choice("emit_x")

    def test_auto_still_allows_at_most_one_call(self):
        assert auto_tool_choice() == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    def test_only_tool_and_any_count_as_forced(self):
        assert is_forced_tool_choice(force_tool_choice("emit_x"))
        assert is_forced_tool_choice({"type": "any"})
        assert not is_forced_tool_choice(auto_tool_choice())
        assert not is_forced_tool_choice(None)


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
