"""Page sections that show a collection, and their check against its fields.

A section names its source as a typed object, so a connected source (a Notion
database, a sheet) can be another ``type`` later without a new section kind.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from backend.copilot.tools.render_ui_models import Medium, Short
from backend.data.collections.model import (
    MAX_CONDITIONS,
    MAX_FIELD_NAME_CHARS,
    AggregateOp,
    Bucket,
    CollectionInfo,
    Condition,
    FieldType,
    Sort,
)

from .design_base import Ref, Strict

FieldName = Annotated[str, Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)]
InputName = Annotated[str, Field(min_length=1, max_length=100)]
MAX_COLLECTIONS_PER_PAGE = 6


class CollectionSource(Strict):
    type: Literal["collection"]
    id: Ref = Field(description="A collection id.")


class _OnCollection(Strict):
    source: CollectionSource


class RowButton(Strict):
    """Runs a workflow on this page with values from the row it sits on."""

    label: Short
    workflow: Ref = Field(description="A workflow with its own section on this page.")
    inputs: dict[InputName, FieldName] = Field(
        min_length=1, max_length=6, description="Workflow input -> the row's field."
    )


class CollectionTableSection(_OnCollection):
    kind: Literal["collection_table"]
    title: Medium | None = None
    columns: list[FieldName] = Field(min_length=1, max_length=8)
    filter: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    sort: Sort | None = None
    editable: bool = Field(
        default=True, description="Cells, add and delete for the owner."
    )
    buttons: list[RowButton] = Field(default_factory=list, max_length=2)


class NumberSpec(Strict):
    label: Short
    op: AggregateOp
    field: FieldName | None = Field(
        default=None, description="A number field; not for count."
    )
    filter: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)


class CollectionNumbersSection(_OnCollection):
    kind: Literal["collection_numbers"]
    numbers: list[NumberSpec] = Field(min_length=1, max_length=4)


class CollectionBoardSection(_OnCollection):
    kind: Literal["collection_board"]
    title: Medium | None = None
    group_by: FieldName = Field(
        description="A select field: its options are the columns, and dragging "
        "a card between them sets it."
    )
    card_title: FieldName
    card_fields: list[FieldName] = Field(default_factory=list, max_length=2)
    filter: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    editable: bool = True
    buttons: list[RowButton] = Field(default_factory=list, max_length=2)


class CollectionCardsSection(_OnCollection):
    kind: Literal["collection_cards"]
    title: Medium | None = None
    card_title: FieldName
    card_fields: list[FieldName] = Field(default_factory=list, max_length=4)
    filter: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    sort: Sort | None = None
    limit: int = Field(default=12, ge=3, le=24)
    buttons: list[RowButton] = Field(default_factory=list, max_length=2)


class CollectionChartSection(_OnCollection):
    kind: Literal["collection_chart"]
    title: Medium | None = None
    chart: Literal["bar", "line"] = "bar"
    value: AggregateOp = "count"
    value_field: FieldName | None = Field(
        default=None, description="A number field, for any value but count."
    )
    group_by: FieldName = Field(description="A select or a date field.")
    bucket: Bucket = Field(default="week", description="For a date: day, week, month.")
    filter: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)


CollectionSection = (
    CollectionTableSection
    | CollectionNumbersSection
    | CollectionBoardSection
    | CollectionCardsSection
    | CollectionChartSection
)
_WITH_BUTTONS = (CollectionTableSection, CollectionBoardSection, CollectionCardsSection)


def section_buttons(section: Any) -> list[RowButton]:
    return section.buttons if isinstance(section, _WITH_BUTTONS) else []


def section_collection(section: Any) -> str | None:
    if isinstance(section, _OnCollection):
        return section.source.id
    return None


def collection_problems(section: Any, collection: CollectionInfo) -> list[str]:
    """Where *section* names something *collection* lacks. A table's missing
    column is not one: the table drops it and still draws."""
    names = _names(collection)
    if isinstance(section, CollectionTableSection):
        if not any(c.casefold() in names for c in section.columns):
            return ["none of the table's columns is a field of the collection"]
        named = [c.field for c in section.filter]
        if section.sort:
            named.append(section.sort.field)
        return _missing(named, names) + _button_fields(section.buttons, names)
    if isinstance(section, CollectionNumbersSection):
        problems = []
        for number in section.numbers:
            problems += _missing([c.field for c in number.filter], names)
            if number.op != "count":
                problems += _number_field(number.label, number.field, names)
        return problems
    if isinstance(section, CollectionBoardSection):
        problems = _missing(
            [
                section.card_title,
                *section.card_fields,
                *[c.field for c in section.filter],
            ],
            names,
        )
        if names.get(section.group_by.casefold()) is not FieldType.SELECT:
            problems.append(
                f"a board groups by a select field; '{section.group_by}' is not one"
            )
        return problems + _button_fields(section.buttons, names)
    if isinstance(section, CollectionCardsSection):
        named = [
            section.card_title,
            *section.card_fields,
            *[c.field for c in section.filter],
        ]
        if section.sort:
            named.append(section.sort.field)
        return _missing(named, names) + _button_fields(section.buttons, names)
    if isinstance(section, CollectionChartSection):
        problems = _missing([c.field for c in section.filter], names)
        if names.get(section.group_by.casefold()) not in (
            FieldType.SELECT,
            FieldType.DATE,
        ):
            problems.append(
                f"a chart groups by a select or a date field; '{section.group_by}' is neither"
            )
        if section.value != "count":
            problems += _number_field("the chart", section.value_field, names)
        return problems
    return []


def _number_field(
    label: str, field: str | None, names: dict[str, FieldType]
) -> list[str]:
    kind = names.get((field or "").casefold())
    if kind is None:
        return [f"{label} names no field the collection has"]
    if kind is not FieldType.NUMBER:
        return [f"{label} needs a number field"]
    return []


def _button_fields(buttons: list[RowButton], names: dict[str, FieldType]) -> list[str]:
    return _missing([f for b in buttons for f in b.inputs.values()], names)


def _names(collection: CollectionInfo) -> dict[str, FieldType]:
    names: dict[str, FieldType] = {}
    for field in collection.fields:
        for name in (field.name, *field.aliases):
            names[name.casefold()] = field.type
    return names


def _missing(named: list[str], names: dict[str, FieldType]) -> list[str]:
    return [
        f"the collection has no field named '{n}'"
        for n in named
        if n.casefold() not in names
    ]
