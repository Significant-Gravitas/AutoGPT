"""Tests for run_mcp_tool's context-size bounding helpers.

MCP servers control tool counts, descriptions, and schemas — every
dimension surfaced to the model must stay bounded or a large catalog can
context-bomb the session.
"""

import json

from backend.copilot.tools.run_mcp_tool import (
    _ERROR_SCHEMA_MAX_CHARS,
    _PARAMS_SUMMARY_MAX_CHARS,
    _bounded_schema_hint,
    _summarize_params,
)


class TestBoundedSchemaHint:
    def test_small_schema_passes_through_verbatim(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        assert json.loads(_bounded_schema_hint(schema)) == schema

    def test_oversized_schema_reduces_to_valid_marked_json(self):
        schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "object",
                    "description": "x" * 300,
                    "properties": {"nested": {"type": "string", "enum": ["v"] * 50}},
                }
                for i in range(40)
            },
            "required": ["field_0"],
        }
        assert len(json.dumps(schema)) > _ERROR_SCHEMA_MAX_CHARS

        hint = _bounded_schema_hint(schema)
        payload = json.loads(hint)  # must be valid JSON, not a sliced string
        assert "$truncated" in payload
        assert payload["required"] == ["field_0"]
        assert set(payload["properties"]) == set(schema["properties"])
        # Nested structure is dropped; per-property entries keep type +
        # shortened description only.
        assert "properties" not in payload["properties"]["field_0"]
        assert len(payload["properties"]["field_0"]["description"]) <= 120
        assert len(hint) < len(json.dumps(schema))


class TestSummarizeParams:
    def test_marks_required_and_caps_length(self):
        schema = {
            "properties": {f"param_{i:03d}": {"type": "string"} for i in range(100)},
            "required": ["param_000"],
        }
        summary = _summarize_params(schema)
        assert summary is not None
        assert summary.startswith("param_000*")
        assert len(summary) <= _PARAMS_SUMMARY_MAX_CHARS + 1  # +ellipsis

    def test_no_properties_returns_none(self):
        assert _summarize_params({}) is None
        assert _summarize_params(None) is None
