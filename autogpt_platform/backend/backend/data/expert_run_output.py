from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import JsonValue

OutputType = Literal["table", "doc", "image", "unknown"]
RunOutputEntry = tuple[datetime | None, datetime, Mapping[str, JsonValue]]

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")
_DOC_MIN_LENGTH = 200


def reconstruct_run_outputs(entries: Iterable[RunOutputEntry]) -> dict[str, list[Any]]:
    ordered = sorted(
        entries,
        key=lambda entry: (entry[0] is None, entry[0] or entry[1]),
    )
    outputs: dict[str, list[Any]] = defaultdict(list)
    for _, _, input_data in ordered:
        name = input_data.get("name")
        if isinstance(name, str):
            outputs[name].append(input_data.get("value"))
    return outputs


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
