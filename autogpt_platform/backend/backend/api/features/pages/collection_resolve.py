"""Collection sections resolved for the page's owner, through their row source."""

import asyncio
from datetime import date
from typing import Any

from backend.copilot.tools.render_ui_models import (
    ChartBlock,
    ChartSeries,
    StatItem,
    StatsBlock,
)
from backend.data.collections.model import (
    Aggregate,
    CollectionField,
    CollectionRowRecord,
    Condition,
    FieldType,
    RowQuery,
)

from .collection_design import (
    CollectionBoardSection,
    CollectionCardsSection,
    CollectionChartSection,
    CollectionNumbersSection,
    CollectionTableSection,
    RowButton,
    section_buttons,
)
from .resolved import (
    BoardColumn,
    PageButton,
    PageCollectionBoard,
    PageCollectionCards,
    PageCollectionChart,
    PageCollectionNumbers,
    PageCollectionTable,
    PageSection,
    PageTableColumn,
)
from .row_sources import RowSource
from .sources import PageSources

TABLE_PAGE_ROWS = 50
BOARD_COLUMN_ROWS = 25
NO_VALUE = "No value"


async def resolve_collection(
    index: int,
    section: Any,
    source: RowSource,
    sources: PageSources,
    user_id: str,
    today: date,
) -> PageSection:
    by_name = _by_name(source.fields)
    buttons = _buttons(section_buttons(section), sources)
    button_fields = {f for b in section_buttons(section) for f in b.inputs.values()}
    keep = {
        by_name[n.casefold()].name for n in button_fields if n.casefold() in by_name
    }
    base = {
        "index": index,
        "collection_id": source.id,
        "collection_name": source.name,
    }
    if isinstance(section, CollectionTableSection):
        return await _table(
            section, source, user_id, today, by_name, keep, buttons, base
        )
    if isinstance(section, CollectionNumbersSection):
        return await _numbers(section, source, user_id, today, by_name, base)
    if isinstance(section, CollectionBoardSection):
        return await _board(
            section, source, user_id, today, by_name, keep, buttons, base
        )
    if isinstance(section, CollectionCardsSection):
        return await _cards(
            section, source, user_id, today, by_name, keep, buttons, base
        )
    assert isinstance(section, CollectionChartSection)
    return await _chart(section, source, user_id, today, by_name, base)


async def _table(
    section: CollectionTableSection,
    source: RowSource,
    user_id: str,
    today: date,
    by_name: dict[str, CollectionField],
    keep: set[str],
    buttons: list[PageButton],
    base: dict[str, Any],
) -> PageCollectionTable:
    shown = [by_name[c.casefold()] for c in section.columns if c.casefold() in by_name]
    page = await source.query(
        user_id,
        RowQuery(
            conditions=section.filter,
            sort=section.sort,
            limit=TABLE_PAGE_ROWS,
            today=today,
        ),
    )
    names = {field.name for field in shown} | keep
    return PageCollectionTable(
        kind="collection_table",
        **base,
        title=section.title,
        columns=[_column(f) for f in shown],
        dropped_columns=[c for c in section.columns if c.casefold() not in by_name],
        rows=[_only(row, names) for row in page.rows],
        total=page.total,
        filter=section.filter,
        sort=section.sort,
        read_only=not (section.editable and source.writable),
        key_field=source.key_field,
        buttons=buttons,
    )


async def _numbers(
    section: CollectionNumbersSection,
    source: RowSource,
    user_id: str,
    today: date,
    by_name: dict[str, CollectionField],
    base: dict[str, Any],
) -> PageCollectionNumbers:
    values = await asyncio.gather(
        *(
            source.aggregate(
                user_id, [Aggregate(op=n.op, field=n.field)], n.filter, today
            )
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
        **base,
        block=StatsBlock(type="stats", items=items),
    )


async def _board(
    section: CollectionBoardSection,
    source: RowSource,
    user_id: str,
    today: date,
    by_name: dict[str, CollectionField],
    keep: set[str],
    buttons: list[PageButton],
    base: dict[str, Any],
) -> PageCollectionBoard:
    group = by_name[section.group_by.casefold()]
    title = by_name[section.card_title.casefold()]
    fields = [by_name[f.casefold()] for f in section.card_fields]
    options: list[str | None] = [*group.options, None]
    pages = await asyncio.gather(
        *(
            source.query(
                user_id,
                RowQuery(
                    conditions=[*section.filter, _in_column(group, option)],
                    limit=BOARD_COLUMN_ROWS,
                    today=today,
                ),
            )
            for option in options
        )
    )
    names = {group.name, title.name, *(f.name for f in fields)} | keep
    columns = [
        BoardColumn(
            option=option,
            rows=[_only(row, names) for row in page.rows],
            total=page.total,
        )
        for option, page in zip(options, pages)
        if option is not None or page.total
    ]
    return PageCollectionBoard(
        kind="collection_board",
        **base,
        title=section.title,
        group_by=group.name,
        card_title=_column(title),
        card_fields=[_column(f) for f in fields],
        columns=columns,
        read_only=not (section.editable and source.writable),
        buttons=buttons,
    )


async def _cards(
    section: CollectionCardsSection,
    source: RowSource,
    user_id: str,
    today: date,
    by_name: dict[str, CollectionField],
    keep: set[str],
    buttons: list[PageButton],
    base: dict[str, Any],
) -> PageCollectionCards:
    title = by_name[section.card_title.casefold()]
    fields = [by_name[f.casefold()] for f in section.card_fields]
    page = await source.query(
        user_id,
        RowQuery(
            conditions=section.filter,
            sort=section.sort,
            limit=section.limit,
            today=today,
        ),
    )
    names = {title.name, *(f.name for f in fields)} | keep
    return PageCollectionCards(
        kind="collection_cards",
        **base,
        title=section.title,
        card_title=_column(title),
        card_fields=[_column(f) for f in fields],
        rows=[_only(row, names) for row in page.rows],
        total=page.total,
        buttons=buttons,
    )


async def _chart(
    section: CollectionChartSection,
    source: RowSource,
    user_id: str,
    today: date,
    by_name: dict[str, CollectionField],
    base: dict[str, Any],
) -> PageCollectionChart:
    group = by_name[section.group_by.casefold()]
    measured = by_name.get((section.value_field or "").casefold())
    groups = await source.group(
        user_id,
        group.name,
        Aggregate(op=section.value, field=measured.name if measured else None),
        bucket=section.bucket if group.type is FieldType.DATE else None,
        conditions=section.filter,
        today=today,
    )
    block = None
    if groups:
        name = "Rows" if section.value == "count" or measured is None else measured.name
        block = ChartBlock(
            type="chart",
            title=section.title,
            kind=section.chart,
            x_labels=[_group_label(g.label, section.bucket, group) for g in groups],
            series=[ChartSeries(name=name, values=[g.value or 0.0 for g in groups])],
            unit=measured.unit if measured and section.value != "count" else None,
        )
    return PageCollectionChart(
        kind="collection_chart", **base, title=section.title, block=block
    )


def _buttons(buttons: list[RowButton], sources: PageSources) -> list[PageButton]:
    resolved = []
    for button in buttons:
        target = sources.workflows.get(button.workflow)
        if target is None:
            continue
        resolved.append(
            PageButton(
                label=button.label,
                library_agent_id=target.id,
                graph_id=target.graph_id,
                graph_version=target.graph_version,
                inputs=dict(button.inputs),
            )
        )
    return resolved


def _in_column(group: CollectionField, option: str | None) -> Condition:
    if option is None:
        return Condition(field=group.name, op="empty")
    return Condition(field=group.name, op="is", value=option)


def _group_label(label: str | None, bucket: str, field: CollectionField) -> str:
    if label is None:
        return NO_VALUE
    if field.type is FieldType.DATE and bucket == "week":
        return f"Week of {label}"
    return label


def _column(field: CollectionField) -> PageTableColumn:
    return PageTableColumn(
        field=field.name, type=field.type, options=field.options, unit=field.unit
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
