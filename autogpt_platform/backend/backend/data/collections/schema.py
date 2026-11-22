"""The owner's changes to a collection's shape: its name, fields and key.

Never a workflow's: a block can only add fields, by writing them.
"""

from typing import Annotated, Any

from prisma.models import Collection, CollectionRow
from pydantic import BaseModel, Field

from backend.data.db import execute_raw_with_schema
from backend.util.json import SafeJson

from .fields import FieldSet, check_field_name, key_text
from .model import (
    MAX_FIELD_NAME_CHARS,
    MAX_FIELDS,
    MAX_NAME_CHARS,
    CollectionError,
    CollectionField,
    FieldSpec,
    FieldType,
)

FieldName = Annotated[str, Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)]


class FieldRename(BaseModel):
    field: FieldName
    to: FieldName


class TypeChange(BaseModel):
    field: FieldName
    type: FieldType


class SchemaChange(BaseModel):
    rename: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    add_fields: list[FieldSpec] = Field(default_factory=list, max_length=MAX_FIELDS)
    remove_fields: list[Annotated[str, Field(max_length=MAX_FIELD_NAME_CHARS)]] = Field(
        default_factory=list, max_length=MAX_FIELDS
    )
    key_field: str | None = Field(default=None, max_length=MAX_FIELD_NAME_CHARS)
    clear_key: bool = False
    rename_fields: list[FieldRename] = Field(
        default_factory=list, max_length=MAX_FIELDS
    )
    change_types: list[TypeChange] = Field(default_factory=list, max_length=MAX_FIELDS)


async def apply_schema_change(
    tx: Any, locked: Collection, change: SchemaChange
) -> Collection:
    fields = FieldSet([CollectionField.model_validate(f) for f in locked.fields])
    key_field = locked.keyField
    for name in change.remove_fields:
        removed = fields.require(name)
        fields.fields.remove(removed)
        fields.changed = True
        await _strip_field(tx, locked.id, removed.name)
        if key_field == removed.name:
            key_field = None
    for rename in change.rename_fields:
        old = fields.require(rename.field)
        new = rename.to.strip()
        check_field_name(new)
        clash = fields.find(new)
        if clash is not None and clash is not old:
            raise CollectionError(f"A field named '{clash.name}' already exists")
        await _rename_in_rows(tx, locked.id, old.name, new)
        old.aliases = [
            a for a in [*old.aliases, old.name] if a.casefold() != new.casefold()
        ]
        if key_field == old.name:
            key_field = new
        old.name = new
        fields.changed = True
    for retype in change.change_types:
        await _retype(tx, locked.id, fields, fields.require(retype.field), retype.type)
    for spec in change.add_fields:
        fields.add(CollectionField(**spec.model_dump()))
    if change.clear_key:
        key_field = None
    if change.key_field:
        key_field = fields.require(change.key_field).name
    if key_field != locked.keyField:
        await _rekey(tx, locked, key_field)
    data: dict[str, Any] = {
        "fields": SafeJson([f.model_dump(mode="json") for f in fields.fields]),
        "keyField": key_field,
    }
    if change.rename:
        clean = " ".join(change.rename.split())
        data |= {"name": clean, "nameKey": clean.casefold()}
    if fields.changed or key_field != locked.keyField:
        data["revision"] = {"increment": 1}
    updated = await Collection.prisma(tx).update(where={"id": locked.id}, data=data)
    assert updated is not None
    return updated


async def _rename_in_rows(tx: Any, collection_id: str, old: str, new: str) -> None:
    moved = (
        'CASE WHEN jsonb_exists("{column}", $2) '
        'THEN ("{column}" - $2) || jsonb_build_object($3::text, "{column}" -> $2) '
        'ELSE "{column}" END'
    )
    columns = ", ".join(
        f'"{c}" = {moved.replace("{column}", c)}'
        for c in ("data", "sourceData", "pending")
    )
    await execute_raw_with_schema(
        f'UPDATE {{schema_prefix}}"CollectionRow" SET {columns}, '
        '"manualFields" = array_replace("manualFields", $2, $3) '
        'WHERE "collectionId" = $1',
        collection_id,
        old,
        new,
        client=tx,
    )


async def _retype(
    tx: Any,
    collection_id: str,
    fields: FieldSet,
    field: CollectionField,
    to: FieldType,
) -> None:
    """Every row's value converted to *to*, refused while any would lose data."""
    if field.type is to:
        return
    target = CollectionField(name=field.name, type=to, unit=field.unit)
    staged = FieldSet([target])
    rows = await CollectionRow.prisma(tx).find_many(
        where={"collectionId": collection_id}
    )
    updates: list[tuple[str, dict[str, Any]]] = []
    failed: list[str] = []
    for row in rows:
        data = row.data if isinstance(row.data, dict) else {}
        if data.get(field.name) is None:
            continue
        try:
            converted = staged.coerce_row({field.name: data[field.name]}, grow=False)
        except CollectionError:
            failed.append(str(data[field.name]))
            continue
        updates.append((row.id, {**data, **converted}))
    if failed:
        raise CollectionError(
            f"{len(failed)} values in '{field.name}' are not {to.value}, "
            f"'{failed[0][:40]}' among them"
        )
    for row_id, data in updates:
        await CollectionRow.prisma(tx).update(
            where={"id": row_id}, data={"data": SafeJson(data)}
        )
    field.type = to
    field.options = staged.require(field.name).options
    fields.changed = True


async def _strip_field(tx: Any, collection_id: str, name: str) -> None:
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"CollectionRow" SET '
        '"data" = "data" - $2, "sourceData" = "sourceData" - $2, '
        '"pending" = "pending" - $2, '
        '"manualFields" = array_remove("manualFields", $2) '
        'WHERE "collectionId" = $1',
        collection_id,
        name,
        client=tx,
    )


async def _rekey(tx: Any, locked: Collection, key_field: str | None) -> None:
    """Every row's key read from *key_field*, refused while two rows share one."""
    rows = await CollectionRow.prisma(tx).find_many(where={"collectionId": locked.id})
    keys = {
        row.id: (
            key_text(row.data.get(key_field))
            if key_field and isinstance(row.data, dict)
            else None
        )
        for row in rows
    }
    seen: dict[str, int] = {}
    for key in keys.values():
        if key is not None:
            seen[key] = seen.get(key, 0) + 1
    shared = [key for key, n in seen.items() if n > 1]
    if shared:
        raise CollectionError(
            f"{key_field} cannot be the key: {len(shared)} values appear on more "
            f"than one row, '{shared[0]}' among them"
        )
    await CollectionRow.prisma(tx).update_many(
        where={"collectionId": locked.id}, data={"key": None}
    )
    for row_id, key in keys.items():
        if key is not None:
            await CollectionRow.prisma(tx).update(
                where={"id": row_id}, data={"key": key}
            )
