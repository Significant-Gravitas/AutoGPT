from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import JsonValue, ValidationError

from backend.data.model import NodeExecutionStats

OutputType = Literal["table", "doc", "image", "unknown"]
RunOutputEntry = tuple[
    datetime | None,
    datetime,
    Mapping[str, JsonValue],
    Mapping[str, list[JsonValue]],
]

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")
_DOC_MIN_LENGTH = 200


def reconstruct_run_outputs(entries: Iterable[RunOutputEntry]) -> dict[str, list[Any]]:
    """Map Agent Output node executions to ``{name: [values]}``, keyed on the
    node's ``name`` input (the block doesn't yield ``name`` when formatting)."""
    ordered = sorted(
        entries,
        key=lambda entry: (entry[0] is None, entry[0] or entry[1]),
    )
    outputs: dict[str, list[Any]] = defaultdict(list)
    for _, _, input_data, output_data in ordered:
        name = input_data.get("name")
        if not isinstance(name, str):
            continue
        outputs[name].extend(output_values(input_data, output_data))
    return outputs


def output_values(
    input_data: Mapping[str, Any], output_data: Mapping[str, list[Any]]
) -> list[Any]:
    """Values an Agent Output node produced: its yielded ``output`` (formatted
    when ``format`` is set), or its input ``value`` if it yielded nothing."""
    return list(output_data.get("output") or [input_data.get("value")])


def moderation_cleared_io(
    raw_stats: object,
) -> tuple[dict[str, Any] | None, dict[str, list[Any]] | None]:
    """A node execution's moderation-cleared ``(inputs, outputs)``, ``None`` for
    a side with nothing cleared. Same precedence as ``NodeExecutionResult.from_db``.
    """
    try:
        stats = NodeExecutionStats.model_validate(raw_stats or {})
    except (ValueError, ValidationError):
        return None, None
    inputs = (
        {
            name: (msgs[-1] if msgs else "")
            for name, msgs in stats.cleared_inputs.items()
        }
        if stats.cleared_inputs
        else None
    )
    outputs = (
        {name: list(msgs) for name, msgs in stats.cleared_outputs.items()}
        if stats.cleared_outputs
        else None
    )
    return inputs, outputs


def classify_output_type(value: object) -> OutputType:
    if isinstance(value, dict):
        return "table"
    if isinstance(value, list) and value:
        if all(isinstance(row, dict) for row in value):
            return "table"
        if all(isinstance(item, str) for item in value):
            strings = [item.strip() for item in value if item.strip()]
            if strings and all(is_image_url(item) for item in strings):
                return "image"
            if len("\n\n".join(strings)) >= _DOC_MIN_LENGTH:
                return "doc"
    if isinstance(value, str):
        stripped = value.strip()
        if is_image_url(stripped):
            return "image"
        if len(stripped) >= _DOC_MIN_LENGTH:
            return "doc"
    return "unknown"


def classify_run_output(
    outputs: Mapping[str, list[Any]],
) -> tuple[OutputType, str | None]:
    for key, values in outputs.items():
        if not values:
            continue
        output_type = classify_output_type(values[0] if len(values) == 1 else values)
        if output_type != "unknown":
            return output_type, key
    return "unknown", None


def is_image_url(value: str) -> bool:
    if not value.lower().startswith("https://"):
        return False
    return urlsplit(value).path.lower().endswith(_IMAGE_EXTENSIONS)
