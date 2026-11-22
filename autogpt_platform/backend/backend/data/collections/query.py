"""SQL for reading rows by filter, search, sort and aggregate.

Built from a collection's stored fields only. Every field name and value is a
bound parameter; the SQL text holds nothing a design, a block input or a
person typed.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any

from .fields import FieldSet, coerce
from .model import (
    Aggregate,
    CollectionError,
    CollectionField,
    Condition,
    FieldType,
    RowQuery,
    Sort,
)

_TEXTUAL = {FieldType.TEXT, FieldType.URL, FieldType.EMAIL, FieldType.SELECT}
_ORDERED = {FieldType.NUMBER, FieldType.DATE}
_SEARCHED = _TEXTUAL | {FieldType.MULTI_SELECT}


class Params:
    def __init__(self, *initial: Any, today: date | None = None):
        self.values: list[Any] = list(initial)
        self.today = today or datetime.now(UTC).date()

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"${len(self.values)}"


def where_clause(fields: FieldSet, query: RowQuery | None, params: Params) -> str:
    """Conditions after the owner and collection, which are $1 and $2."""
    parts = ['r."userId" = $1', 'r."collectionId" = $2']
    if query is None:
        return " AND ".join(parts)
    parts += [_condition(fields, c, params) for c in query.conditions]
    if query.search and query.search.strip():
        searched = [f for f in fields.fields if f.type in _SEARCHED]
        if searched:
            pattern = params.add(f"%{_escape_like(query.search.strip())}%")
            parts.append(
                "("
                + " OR ".join(
                    f"{_text(params.add(f.name))} ILIKE {pattern} ESCAPE '\\'"
                    for f in searched
                )
                + ")"
            )
        else:
            parts.append("false")
    for name, options in query.chips.items():
        field = fields.require(name)
        if not options:
            continue
        chosen = [_equals(field, option, params) for option in options[:20]]
        parts.append("(" + " OR ".join(chosen) + ")")
    return " AND ".join(parts)


def where_conditions(
    fields: FieldSet, conditions: list[Condition], params: Params
) -> str:
    """*conditions* alone, for an aggregate, which takes no search or chips."""
    return where_clause(fields, RowQuery(conditions=conditions), params)


def order_clause(fields: FieldSet, sort: Sort | None, params: Params) -> str:
    tie = 'r."createdAt" DESC, r."id" ASC'
    if sort is None:
        return tie
    field = fields.require(sort.field)
    direction = "DESC" if sort.direction == "desc" else "ASC"
    return f"{_sortable(field, params)} {direction} NULLS LAST, {tie}"


def aggregate_expr(fields: FieldSet, aggregate: Aggregate, params: Params) -> str:
    if aggregate.op == "count":
        return "count(*)::float8"
    if not aggregate.field:
        raise CollectionError(f"A {aggregate.op} names a number field")
    field = fields.require(aggregate.field)
    if field.type is not FieldType.NUMBER:
        raise CollectionError(
            f"A {aggregate.op} needs a number field; '{field.name}' is {field.type.value}"
        )
    function = {"sum": "sum", "average": "avg", "min": "min", "max": "max"}[
        aggregate.op
    ]
    return f"{function}({_number(params.add(field.name))})::float8"


def group_expr(field: CollectionField, bucket: str | None, params: Params) -> str:
    """The label rows are grouped under: a select's option, or a date's day,
    week (its Monday) or month."""
    name = params.add(field.name)
    if field.type is FieldType.SELECT:
        return _text(name)
    if field.type is not FieldType.DATE:
        raise CollectionError(
            f"A chart groups by a select or a date field; '{field.name}' is "
            f"{field.type.value}"
        )
    day = f"left({_text(name)}, 10)"
    if bucket == "month":
        return f"left({day}, 7)"
    if bucket == "week":
        return f"to_char(date_trunc('week', ({day})::date), 'YYYY-MM-DD')"
    return day


def _condition(fields: FieldSet, condition: Condition, params: Params) -> str:
    field = fields.require(condition.field)
    op = condition.op
    if op in ("empty", "not_empty"):
        empty = _is_empty(field, params.add(field.name))
        return empty if op == "empty" else f"NOT {empty}"
    if op in ("today", "before_today", "last_days", "next_days", "this_month"):
        return _relative(field, op, condition.value, params)
    if condition.value is None:
        raise CollectionError(f"'{op}' on '{field.name}' needs a value")
    if op == "is":
        return _equals(field, condition.value, params)
    if op == "is_not":
        return f"NOT coalesce({_equals(field, condition.value, params)}, false)"
    if op == "contains":
        return _contains(field, condition.value, params)
    return _compare(field, op, condition.value, params)


def _equals(field: CollectionField, value: Any, params: Params) -> str:
    name = params.add(field.name)
    if field.type is FieldType.NUMBER:
        return f"{_number(name)} = {params.add(str(_value(field, value)))}::numeric"
    if field.type is FieldType.CHECKBOX:
        flag = "true" if _value(field, value) else "false"
        return f"{_bool(name)} = {params.add(flag)}::boolean"
    if field.type is FieldType.DATE:
        day = _value(field, value)
        if len(day) == 10:
            return f"left({_text(name)}, 10) = {params.add(day)}"
        return f"{_text(name)} = {params.add(day)}"
    if field.type is FieldType.MULTI_SELECT:
        return _has_option(name, _one_option(field, value), params)
    return f"lower({_text(name)}) = lower({params.add(str(value).strip())})"


def _contains(field: CollectionField, value: Any, params: Params) -> str:
    name = params.add(field.name)
    if field.type is FieldType.MULTI_SELECT:
        return _has_option(name, _one_option(field, value), params)
    if field.type not in _TEXTUAL:
        raise CollectionError(f"'contains' does not apply to {field.type.value} fields")
    pattern = params.add(f"%{_escape_like(str(value).strip())}%")
    return f"{_text(name)} ILIKE {pattern} ESCAPE '\\'"


def _compare(field: CollectionField, op: str, value: Any, params: Params) -> str:
    if field.type not in _ORDERED:
        raise CollectionError(
            f"'{op}' compares numbers and dates; '{field.name}' is {field.type.value}"
        )
    sign = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[op]
    name = params.add(field.name)
    converted = _value(field, value)
    if field.type is FieldType.NUMBER:
        return f"{_number(name)} {sign} {params.add(str(converted))}::numeric"
    if len(converted) == 10:
        return f"left({_text(name)}, 10) {sign} {params.add(converted)}"
    return f"{_text(name)} {sign} {params.add(converted)}"


def _relative(field: CollectionField, op: str, value: Any, params: Params) -> str:
    if field.type is not FieldType.DATE:
        raise CollectionError(
            f"'{op}' applies to date fields; '{field.name}' is {field.type.value}"
        )
    day = f"left({_text(params.add(field.name))}, 10)"
    today = params.today
    if op == "today":
        return f"{day} = {params.add(today.isoformat())}"
    if op == "before_today":
        return f"{day} < {params.add(today.isoformat())}"
    if op == "this_month":
        return f"left({day}, 7) = {params.add(today.isoformat()[:7])}"
    days = timedelta(days=_day_count(op, value))
    start, end = (today - days, today) if op == "last_days" else (today, today + days)
    return f"{day} BETWEEN {params.add(start.isoformat())} AND {params.add(end.isoformat())}"


def _day_count(op: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CollectionError(f"'{op}' takes a number of days")
    try:
        days = int(float(value))
    except ValueError:
        raise CollectionError(f"'{op}' takes a number of days") from None
    if not 1 <= days <= 366:
        raise CollectionError(f"'{op}' takes 1 to 366 days")
    return days


def _is_empty(field: CollectionField, name: str) -> str:
    if field.type is FieldType.MULTI_SELECT:
        return (
            f'coalesce(jsonb_array_length(CASE WHEN jsonb_typeof(r."data" -> {name})'
            f" = 'array' THEN r.\"data\" -> {name} END), 0) = 0"
        )
    if field.type is FieldType.CHECKBOX:
        return f"jsonb_typeof(r.\"data\" -> {name}) IS DISTINCT FROM 'boolean'"
    return f"coalesce({_text(name)}, '') = ''"


def _has_option(name: str, option: str, params: Params) -> str:
    return (
        f"coalesce(r.\"data\" -> {name}, '[]'::jsonb) @> "
        f"jsonb_build_array({params.add(option)}::text)"
    )


def _sortable(field: CollectionField, params: Params) -> str:
    name = params.add(field.name)
    if field.type is FieldType.NUMBER:
        return _number(name)
    if field.type is FieldType.CHECKBOX:
        return _bool(name)
    if field.type is FieldType.DATE:
        return _text(name)
    return f"lower({_text(name)})"


def _value(field: CollectionField, value: Any) -> Any:
    try:
        return coerce(field, value)
    except (TypeError, ValueError) as e:
        raise CollectionError(f"Filter on '{field.name}': {e}") from e


def _one_option(field: CollectionField, value: Any) -> str:
    text = str(value).strip()
    return next((o for o in field.options if o.casefold() == text.casefold()), text)


def _text(name: str) -> str:
    return f'(r."data" ->> {name})'


def _number(name: str) -> str:
    return (
        f"(CASE WHEN jsonb_typeof(r.\"data\" -> {name}) = 'number' "
        f'THEN (r."data" ->> {name})::numeric END)'
    )


def _bool(name: str) -> str:
    return (
        f"(CASE WHEN jsonb_typeof(r.\"data\" -> {name}) = 'boolean' "
        f'THEN (r."data" ->> {name})::boolean ELSE false END)'
    )


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
