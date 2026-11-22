"""Collection sections resolved for the page's owner, through their row source."""

import asyncio
from typing import Any

from backend.copilot.tools.render_ui_models import StatItem, StatsBlock
from backend.data.collections.model import (
    Aggregate,
    CollectionField,
    CollectionRowRecord,
    FieldType,
    RowQuery,
)

from .collection_design import CollectionNumbersSection, CollectionTableSection
from .resolved import PageCollectionNumbers, PageCollectionTable, PageTableColumn
from .row_sources import RowSource

TABLE_PAGE_ROWS = 50


async def resolve_table(
    index: int, section: CollectionTableSection, source: RowSource, user_id: str
) -> PageCollectionTable:
    by_name = _by_name(source.fields)
    shown = [by_name[c.casefold()] for c in section.columns if c.casefold() in by_name]
    page = await source.query(
        user_id,
        RowQuery(conditions=section.filter, sort=section.sort, limit=TABLE_PAGE_ROWS),
    )
    names = {field.name for field in shown}
    return PageCollectionTable(
        kind="collection_table",
        index=index,
        collection_id=source.id,
        collection_name=source.name,
        title=section.title,
        columns=[
            PageTableColumn(field=f.name, type=f.type, options=f.options, unit=f.unit)
            for f in shown
        ],
        dropped_columns=[c for c in section.columns if c.casefold() not in by_name],
        rows=[_only(row, names) for row in page.rows],
        total=page.total,
        filter=section.filter,
        sort=section.sort,
        read_only=not (section.editable and source.writable),
        key_field=source.key_field,
    )


async def resolve_numbers(
    index: int, section: CollectionNumbersSection, source: RowSource, user_id: str
) -> PageCollectionNumbers:
    by_name = _by_name(source.fields)
    values = await asyncio.gather(
        *(
            source.aggregate(user_id, [Aggregate(op=n.op, field=n.field)], n.filter)
            for n in section.numbers
        )
    )
    items = [
        StatItem(
            label=number.label,
            value=_number_text(
                value, by_name.get((number.field or "").casefold()), number.op
            ),
        )
        for number, (value,) in zip(section.numbers, values)
    ]
    return PageCollectionNumbers(
        kind="collection_numbers",
        index=index,
        collection_id=source.id,
        collection_name=source.name,
        block=StatsBlock(type="stats", items=items),
    )


def _by_name(fields: list[CollectionField]) -> dict[str, CollectionField]:
    found: dict[str, CollectionField] = {}
    for field in fields:
        for name in (field.name, *field.aliases):
            found[name.casefold()] = field
    return found


def _only(row: CollectionRowRecord, names: set[str]) -> CollectionRowRecord:
    return row.model_copy(
        update={
            "values": {k: v for k, v in row.values.items() if k in names},
            "pending": {k: v for k, v in row.pending.items() if k in names},
        }
    )


def _number_text(value: float | None, field: CollectionField | None, op: str) -> str:
    if value is None:
        return "–"
    if op == "count":
        return f"{int(value):,}"
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    unit = field.unit if field and field.type is FieldType.NUMBER else None
    return _with_unit(text, unit)


def _with_unit(text: str, unit: Any) -> str:
    if not unit:
        return text
    return (
        f"{unit}{text}" if len(unit) == 1 and not unit.isalnum() else f"{text} {unit}"
    )
