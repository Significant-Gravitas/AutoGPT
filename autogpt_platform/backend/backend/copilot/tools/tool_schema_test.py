"""Schema regression tests for all registered CoPilot tools.

Validates that every tool in TOOL_REGISTRY produces a well-formed schema:
- description is non-empty
- all `required` fields exist in `properties`
- every property has a `type` and `description`
- total schema character budget does not regress past threshold
"""

import json
from typing import Any, cast

import pytest

from backend.copilot.tools import TOOL_REGISTRY

# Character budget (~4 chars/token heuristic, targeting ~8000 tokens).
# Bumped 32000 -> 32500 on PR #12699 to fit two pieces of load-bearing
# guidance: the wait_for_result dispatch-mode docs on run_agent
# (tells the LLM when to block vs fire-and-forget, and what each
# response shape carries) and the dry_run description. Keeps the
# regression gate effective while accepting a deliberate ~120-token
# spend on LLM-decision-critical copy.
_CHAR_BUDGET = 32_500


@pytest.fixture(scope="module")
def all_tool_schemas() -> list[tuple[str, Any]]:
    """Return (tool_name, openai_schema) pairs for every registered tool."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


def _get_parametrize_data() -> list[tuple[str, object]]:
    """Build parametrize data at collection time."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


@pytest.mark.parametrize(
    "tool_name,schema",
    _get_parametrize_data(),
    ids=[name for name, _ in _get_parametrize_data()],
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


def test_browser_act_action_enum_complete() -> None:
    """Assert browser_act action enum still contains all 14 supported actions.

    This prevents future PRs from accidentally dropping actions during description
    trimming. The enum is the authoritative list — this locks it at 14 values.
    """
    tool = TOOL_REGISTRY["browser_act"]
    schema = tool.as_openai_tool()
    fn_def = schema["function"]
    params = cast(dict[str, Any], fn_def.get("parameters", {}))
    actions = params["properties"]["action"]["enum"]
    expected = {
        "click",
        "dblclick",
        "fill",
        "type",
        "scroll",
        "hover",
        "press",
        "check",
        "uncheck",
        "select",
        "wait",
        "back",
        "forward",
        "reload",
    }
    assert set(actions) == expected, (
        f"browser_act action enum changed. Got {set(actions)}, expected {expected}. "
        "If you added/removed an action, update this test intentionally."
    )


def test_total_schema_char_budget() -> None:
    """Assert total tool schema size stays under the character budget.

    This locks in the 34% token reduction from #12398 and prevents future
    description bloat from eroding the gains. Uses character count with a
    ~4 chars/token heuristic (budget of 32000 chars ≈ 8000 tokens).
    Character count is tokenizer-agnostic — no dependency on GPT or Claude
    tokenizers — while still providing a stable regression gate.
    """
    schemas = [tool.as_openai_tool() for tool in TOOL_REGISTRY.values()]
    serialized = json.dumps(schemas)
    total_chars = len(serialized)
    assert total_chars < _CHAR_BUDGET, (
        f"Tool schemas use {total_chars} chars (~{total_chars // 4} tokens), "
        f"exceeding budget of {_CHAR_BUDGET} chars (~{_CHAR_BUDGET // 4} tokens). "
        f"Description bloat detected — trim descriptions or raise the budget intentionally."
    )
