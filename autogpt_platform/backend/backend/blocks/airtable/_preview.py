"""Airtable tables and fields as the approval card shows them: the names the
user will see in their base, and each field's type in plain words."""

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
    kind = _KINDS.get(raw) or re.sub(r"(?<!^)(?=[A-Z])", " ", raw).capitalize()
    choices = _choice_names(options)
    return f"{kind} of {', '.join(choices)}" if choices else kind


def table_preview(name: Any, fields: Any) -> PreviewTable:
    return PreviewTable(
        name=str(name or ""),
        fields=[
            PreviewField(
                name=str(field.get("name") or ""),
                kind=field_kind(field.get("type"), field.get("options")),
            )
            for field in _dicts(fields)
        ],
    )


def tables_preview(tables: Any) -> list[PreviewTable]:
    return [table_preview(t.get("name"), t.get("fields")) for t in _dicts(tables)]


def preview(
    ask: str, object: Any, tables: list[PreviewTable], covers: list[str]
) -> ChangePreview:
    return ChangePreview(
        ask=ask,
        object=str(object) if object else None,
        tables=tables,
        covers=covers,
    )


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
