"""Shared helpers for chat tools."""

from typing import Any


def get_inputs_from_schema(
    input_schema: dict[str, Any],
    exclude_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract input field info from JSON schema."""
    if not isinstance(input_schema, dict):
        return []

    exclude = exclude_fields or set()
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    return [
        {
            "name": name,
            "title": schema.get("title", name),
            "type": schema.get("type", "string"),
            "description": schema.get("description", ""),
            "required": name in required,
            "default": schema.get("default"),
        }
        for name, schema in properties.items()
        if name not in exclude
    ]
