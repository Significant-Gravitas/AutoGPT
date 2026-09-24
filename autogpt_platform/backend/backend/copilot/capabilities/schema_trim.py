"""Schema trimming for ``describe_capability``: large enums are the single
biggest schema cost (the LLM ``model`` enum alone is ~110 values), so they
are collapsed to a sample unless the model asks to ``expand``."""

from typing import Any

ENUM_COLLAPSE_LIMIT = 12
ENUM_SAMPLE = 8


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
