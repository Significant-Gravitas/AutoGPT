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
    CollectionInfo,
    Condition,
    FieldType,
    Sort,
)

from .design_base import Ref, Strict

FieldName = Annotated[str, Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)]
MAX_COLLECTIONS_PER_PAGE = 6


class CollectionSource(Strict):
    type: Literal["collection"]
    id: Ref = Field(description="A collection id.")


class _OnCollection(Strict):
    source: CollectionSource


class CollectionTableSection(_OnCollection):
    kind: Literal["collection_table"]
    title: Medium | None = None
    columns: list[FieldName] = Field(min_length=1, max_length=8)
    filter: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    sort: Sort | None = None
    editable: bool = Field(
        default=True, description="Cells, add and delete for the owner."
    )


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


CollectionSection = CollectionTableSection | CollectionNumbersSection


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
        return _missing(named, names)
    if isinstance(section, CollectionNumbersSection):
        problems = []
        for number in section.numbers:
            problems += _missing([c.field for c in number.filter], names)
            if number.op == "count":
                continue
            field = names.get((number.field or "").casefold())
            if field is None:
                problems.append(f"'{number.label}' names no field the collection has")
            elif field is not FieldType.NUMBER:
                problems.append(f"'{number.label}' needs a number field")
        return problems
    return []


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
