"""Airtable tables and fields as the approval card shows them: the names the
user will see in their base, and each field's type in plain words."""

import json
import re
from typing import Any

from backend.sdk import ChangePreview, PreviewField, PreviewTable

_KINDS = {
    "singleLineText": "Text",
    "multilineText": "Long text",
    "richText": "Formatted text",
    "email": "Email",
    "url": "Web link",
    "phoneNumber": "Phone number",
    "number": "Number",
    "percent": "Percent",
    "currency": "Money",
    "duration": "Duration",
    "rating": "Rating",
    "checkbox": "Checkbox",
    "singleSelect": "Choice",
    "multipleSelects": "Choices",
    "singleCollaborator": "Person",
    "multipleCollaborators": "People",
    "multipleRecordLinks": "Link to another table",
    "date": "Date",
    "dateTime": "Date and time",
    "multipleAttachments": "Files",
    "barcode": "Barcode",
    "button": "Button",
    "autoNumber": "Automatic number",
    "formula": "Formula",
    "rollup": "Summary of linked rows",
    "count": "Count of linked rows",
    "lookup": "Value from linked rows",
    "multipleLookupValues": "Value from linked rows",
    "createdTime": "Created time",
    "lastModifiedTime": "Last changed time",
    "createdBy": "Created by",
    "lastModifiedBy": "Last changed by",
    "aiText": "AI text",
}


def field_kind(field_type: Any, options: Any = None) -> str:
    """``currency`` is "Money"; a select also names its choices."""
    raw = str(getattr(field_type, "value", field_type) or "")
    kind = _KINDS.get(raw) or _words(raw).capitalize()
    choices = _choice_names(options)
    return f"{kind} of {', '.join(choices)}" if choices else kind


def table_preview(name: Any, fields: Any, description: Any = "") -> PreviewTable:
    return PreviewTable(
        name=str(name or ""),
        description=str(description or ""),
        fields=[
            PreviewField(
                name=str(field.get("name") or ""),
                kind=field_kind(field.get("type"), field.get("options")),
                detail=field_detail(field.get("description"), field.get("options")),
            )
            for field in _dicts(fields)
        ],
    )


def tables_preview(tables: Any) -> list[PreviewTable]:
    return [
        table_preview(t.get("name"), t.get("fields"), t.get("description"))
        for t in _dicts(tables)
    ]


def field_detail(description: Any, options: Any) -> str:
    """The field's note, then each setting its kind doesn't already say, in
    words: ``{"precision": 2}`` reads "precision 2"."""
    if not isinstance(options, dict):
        options = {}
    settings = [
        *_choice_settings(options.get("choices")),
        *(
            f"{_words(key)} {_value(value)}"
            for key, value in options.items()
            if key != "choices"
        ),
    ]
    return " · ".join([str(description), *settings] if description else settings)


def preview(
    ask: str, object: Any, tables: list[PreviewTable], covers: list[str]
) -> ChangePreview:
    return ChangePreview(
        ask=ask,
        object=str(object) if object else None,
        tables=tables,
        covers=covers,
    )


def _choice_settings(choices: Any) -> list[str]:
    """A choice's name is in the kind; anything else about it (a color) is
    said here: "New: color greenBright"."""
    return [
        f"{choice.get('name')}: "
        + ", ".join(
            f"{_words(k)} {_value(v)}" for k, v in choice.items() if k != "name"
        )
        for choice in _dicts(choices)
        if set(choice) - {"name"}
    ]


def _words(key: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", str(key)).lower()


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(", ", ": "))
    return str(value)


def _dicts(value: Any) -> list[dict]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _choice_names(options: Any) -> list[str]:
    if not isinstance(options, dict):
        return []
    return [str(c["name"]) for c in _dicts(options.get("choices")) if c.get("name")]
