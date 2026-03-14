"""Schema regression tests for all registered CoPilot tools.

Validates that every tool in TOOL_REGISTRY produces a well-formed schema:
- description is non-empty
- all `required` fields exist in `properties`
- every property has a `type` and `description`
"""

import pytest

from backend.copilot.tools import TOOL_REGISTRY


def _get_all_tool_schemas() -> list[tuple[str, object]]:
    """Return (tool_name, openai_schema) pairs for every registered tool."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


@pytest.mark.parametrize(
    "tool_name,schema",
    _get_all_tool_schemas(),
    ids=[name for name, _ in _get_all_tool_schemas()],
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
