"""Schema regression tests for all registered CoPilot tools.

Validates that every tool in TOOL_REGISTRY produces a well-formed schema:
- description is non-empty
- all `required` fields exist in `properties`
- every property has a `type` and `description`
- total token budget does not regress past 8000 tokens
"""

import json

import pytest
import tiktoken

from backend.copilot.tools import TOOL_REGISTRY

_TOKEN_BUDGET = 8_000


def _get_all_tool_schemas() -> list[tuple[str, object]]:
    """Return (tool_name, openai_schema) pairs for every registered tool."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


_ALL_SCHEMAS = _get_all_tool_schemas()


@pytest.mark.parametrize(
    "tool_name,schema",
    _ALL_SCHEMAS,
    ids=[name for name, _ in _ALL_SCHEMAS],
)
class TestToolSchema:
    """Validate schema invariants for every registered tool."""

    def test_description_non_empty(self, tool_name: str, schema: dict) -> None:
        desc = schema["function"].get("description", "")
        assert desc, f"Tool '{tool_name}' has an empty description"

    def test_required_fields_exist_in_properties(
        self, tool_name: str, schema: dict
    ) -> None:
        params = schema["function"].get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        for field in required:
            assert field in properties, (
                f"Tool '{tool_name}': required field '{field}' "
                f"not found in properties {list(properties.keys())}"
            )

    def test_every_property_has_type_and_description(
        self, tool_name: str, schema: dict
    ) -> None:
        params = schema["function"].get("parameters", {})
        properties = params.get("properties", {})
        for prop_name, prop_def in properties.items():
            assert (
                "type" in prop_def
            ), f"Tool '{tool_name}', property '{prop_name}' is missing 'type'"
            assert (
                "description" in prop_def
            ), f"Tool '{tool_name}', property '{prop_name}' is missing 'description'"


def test_total_schema_token_budget() -> None:
    """Assert total tool schema size stays under the token budget.

    This locks in the 34% token reduction from #12398 and prevents future
    description bloat from eroding the gains. Budget is set to 8000 tokens.
    Note: this measures tool JSON only (not the full system prompt); the actual
    baseline for tool schemas alone is ~6470 tokens, giving ~19% headroom.
    Uses cl100k_base (GPT-4 tokenizer) as a proxy — actual Claude token counts
    will differ slightly, but it's a stable regression gate.
    """
    schemas = [tool.as_openai_tool() for tool in TOOL_REGISTRY.values()]
    serialized = json.dumps(schemas)
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = len(enc.encode(serialized))
    assert total_tokens < _TOKEN_BUDGET, (
        f"Tool schemas use {total_tokens} tokens, exceeding budget of {_TOKEN_BUDGET}. "
        f"Description bloat detected — trim descriptions or raise the budget intentionally."
    )
