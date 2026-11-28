"""Field types: what fits each one, what converts without loss, and what a new
field's type is inferred from."""

import json
import math
import re
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlparse

from .model import (
    MAX_FIELD_NAME_CHARS,
    MAX_FIELDS,
    MAX_OPTIONS,
    MAX_ROW_BYTES,
    MAX_TEXT_CHARS,
    CollectionError,
    CollectionField,
    FieldType,
)

_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0", ""}


class FieldSet:
    """A collection's fields, found by name or alias without case, and grown
    as writers name new ones."""

    def __init__(self, fields: list[CollectionField]):
        self.fields = [f.model_copy(deep=True) for f in fields]
        self.changed = False

    def find(self, name: str) -> CollectionField | None:
        wanted = name.strip().casefold()
        for field in self.fields:
            if field.name.casefold() == wanted or wanted in (
                a.casefold() for a in field.aliases
            ):
                return field
        return None

    def require(self, name: str) -> CollectionField:
        field = self.find(name)
        if field is None:
            raise CollectionError(f"This collection has no field named '{name}'")
        return field

    def add(self, field: CollectionField) -> CollectionField:
        check_field_name(field.name)
        if self.find(field.name):
            raise CollectionError(f"A field named '{field.name}' already exists")
        if len(self.fields) >= MAX_FIELDS:
            raise CollectionError(
                f"A collection holds at most {MAX_FIELDS} fields; '{field.name}' "
                "would be one more"
            )
        self.fields.append(field)
        self.changed = True
        return field

    def coerce_row(self, values: dict[str, Any], *, grow: bool) -> dict[str, Any]:
        """*values* keyed by canonical field name and converted to each type;
        with *grow*, an unknown name becomes a field of the inferred type."""
        row: dict[str, Any] = {}
        for name, value in values.items():
            field = self.find(name)
            if field is None:
                if not grow:
                    raise CollectionError(
                        f"This collection has no field named '{name}'"
                    )
                field = self.add(
                    CollectionField(name=name.strip(), type=infer_type(value))
                )
            row[field.name] = self._coerce(field, value)
        return row

    def _coerce(self, field: CollectionField, value: Any) -> Any:
        try:
            converted = coerce(field, value)
        except (TypeError, ValueError) as e:
            raise CollectionError(f"Field '{field.name}': {e}") from e
        if field.type in (FieldType.SELECT, FieldType.MULTI_SELECT):
            converted = self._with_options(field, converted)
        return converted

    def _with_options(self, field: CollectionField, value: Any) -> Any:
        picked = value if isinstance(value, list) else [value]
        resolved = []
        for option in picked:
            if option is None:
                continue
            known = next(
                (o for o in field.options if o.casefold() == option.casefold()), None
            )
            if known is None:
                if len(field.options) >= MAX_OPTIONS:
                    raise CollectionError(
                        f"Field '{field.name}' has the most options a field takes "
                        f"({MAX_OPTIONS}); '{option}' is not one of them"
                    )
                field.options.append(option)
                self.changed = True
                known = option
            resolved.append(known)
        if isinstance(value, list):
            return list(dict.fromkeys(resolved))
        return resolved[0] if resolved else None


def coerce(field: CollectionField, value: Any) -> Any:
    """*value* as *field*'s type, converted only where nothing is lost; None clears."""
    if value is None:
        return None
    if field.type is FieldType.MULTI_SELECT:
        items = value if isinstance(value, list) else [value]
        return [_option(item) for item in items if item is not None]
    return _COERCERS[field.type](value)


def infer_type(value: Any) -> FieldType:
    if isinstance(value, bool):
        return FieldType.CHECKBOX
    if isinstance(value, (int, float)):
        return FieldType.NUMBER
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return FieldType.MULTI_SELECT
    if isinstance(value, str):
        text = value.strip()
        if _as_date(text) is not None:
            return FieldType.DATE
        if _is_http_url(text):
            return FieldType.URL
    return FieldType.TEXT


def check_field_name(name: str) -> None:
    clean = name.strip()
    if not clean or len(clean) > MAX_FIELD_NAME_CHARS:
        raise CollectionError(f"A field name is 1 to {MAX_FIELD_NAME_CHARS} characters")
    if clean.startswith("_"):
        raise CollectionError("A field name cannot start with '_'")


def check_row_size(values: dict[str, Any]) -> None:
    size = len(json.dumps(values, ensure_ascii=False).encode())
    if size > MAX_ROW_BYTES:
        raise CollectionError(
            f"A row holds at most {MAX_ROW_BYTES // 1024} KB; this one is "
            f"{math.ceil(size / 1024)} KB"
        )


def key_text(value: Any) -> str | None:
    """A key field's value as the text the row is matched on."""
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        raise ValueError("a list or an object is not text")
    text = value if isinstance(value, str) else _plain(value)
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is at most {MAX_TEXT_CHARS:,} characters")
    return text


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError("a checkbox value is not a number")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("not a finite number")
        return value
    if isinstance(value, str) and _NUMBER.fullmatch(value.strip()):
        number = float(value.strip())
        return int(number) if number.is_integer() and "." not in value else number
    raise ValueError(f"'{_short(value)}' is not a number")


def _date(value: Any) -> str:
    if not isinstance(value, str) or (parsed := _as_date(value.strip())) is None:
        raise ValueError(
            f"'{_short(value)}' is not a date; use 2026-11-22 or 2026-11-22T14:00"
        )
    return parsed


def _checkbox(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in _TRUE | _FALSE:
        return value.strip().casefold() in _TRUE
    raise ValueError(f"'{_short(value)}' is not true or false")


def _url(value: Any) -> str:
    if not isinstance(value, str) or not _is_http_url(value.strip()):
        raise ValueError(f"'{_short(value)}' is not an http or https address")
    return _text(value.strip())


def _email(value: Any) -> str:
    if not isinstance(value, str) or not _EMAIL.fullmatch(value.strip()):
        raise ValueError(f"'{_short(value)}' is not an email address")
    return _text(value.strip())


def _option(value: Any) -> str:
    if isinstance(value, (dict, list, bool)):
        raise ValueError(f"'{_short(value)}' is not an option")
    text = _plain(value) if not isinstance(value, str) else value.strip()
    if not text or len(text) > 100:
        raise ValueError("an option is 1 to 100 characters")
    return text


_COERCERS = {
    FieldType.TEXT: _text,
    FieldType.NUMBER: _number,
    FieldType.DATE: _date,
    FieldType.SELECT: _option,
    FieldType.CHECKBOX: _checkbox,
    FieldType.URL: _url,
    FieldType.EMAIL: _email,
}


def _as_date(text: str) -> str | None:
    if len(text) < 10 or not text[:4].isdigit():
        return None
    try:
        if len(text) == 10:
            return date.fromisoformat(text).isoformat()
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    moment = moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_http_url(text: str) -> bool:
    if any(c.isspace() for c in text):
        return False
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _plain(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= 40 else text[:39] + "…"
