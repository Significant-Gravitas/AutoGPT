"""Schema trimming for ``describe_capability``: large enums (the LLM ``model``
enum alone is ~110 values) and deeply nested output models (one RMFG quote
output is ~20k characters) are the biggest schema costs, so both are
collapsed unless the model asks to ``expand``."""

from typing import Any

ENUM_COLLAPSE_LIMIT = 12
ENUM_SAMPLE = 8
# Levels of ``properties`` an output schema keeps: the output pins, then the
# fields of each pin.  Objects nested below that are listed by field name.
OUTPUT_FIELD_DEPTH = 2
# Keywords whose value is data rather than a schema, so it is never rewritten.
_DATA_KEYWORDS = frozenset({"default", "examples", "example", "const", "enum"})


def collapse_large_enums(node: Any, limit: int = ENUM_COLLAPSE_LIMIT) -> Any:
    """Return *node* with every ``enum`` longer than *limit* replaced by its
    first ``ENUM_SAMPLE`` values plus a note carrying the full count."""
    if isinstance(node, list):
        return [collapse_large_enums(item, limit) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "enum" and isinstance(value, list) and len(value) > limit:
            cleaned["enum_sample"] = value[:ENUM_SAMPLE]
            cleaned["enum_count"] = len(value)
            cleaned["enum_note"] = (
                f"{len(value)} allowed values; {ENUM_SAMPLE} shown. "
                "Call describe_capability with expand=true for all of them."
            )
            continue
        cleaned[key] = collapse_large_enums(value, limit)
    return cleaned


def collapse_nested_objects(node: Any, depth: int = OUTPUT_FIELD_DEPTH) -> Any:
    """Return *node* with every object schema below *depth* levels of
    ``properties`` reduced to the names of its fields."""
    if isinstance(node, list):
        return [collapse_nested_objects(item, depth) for item in node]
    if not isinstance(node, dict):
        return node
    properties = node.get("properties")
    collapse = depth <= 0 and isinstance(properties, dict)
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DATA_KEYWORDS:
            cleaned[key] = value
        elif key == "properties" and isinstance(value, dict):
            if not collapse:
                cleaned[key] = {
                    name: collapse_nested_objects(sub, depth - 1)
                    for name, sub in value.items()
                }
        elif not (collapse and key == "required"):
            # A collapsed object's other schemas (``additionalProperties``,
            # ``items``, ``anyOf``) are trimmed too, not kept whole.
            cleaned[key] = collapse_nested_objects(value, depth)
    if collapse:
        cleaned["field_names"] = list(properties)
    return cleaned
