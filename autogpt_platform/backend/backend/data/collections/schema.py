"""The owner's changes to a collection's shape: its name, fields and key.

Never a workflow's: a block can only add fields, by writing them.
"""

from typing import Annotated, Any

from prisma.models import Collection, CollectionRow
from pydantic import BaseModel, Field

from backend.data.db import execute_raw_with_schema
from backend.util.json import SafeJson

from .fields import FieldSet, key_text
from .model import (
    MAX_FIELD_NAME_CHARS,
    MAX_FIELDS,
    MAX_NAME_CHARS,
    CollectionError,
    CollectionField,
    FieldSpec,
)


class SchemaChange(BaseModel):
    rename: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    add_fields: list[FieldSpec] = Field(default_factory=list, max_length=MAX_FIELDS)
    remove_fields: list[Annotated[str, Field(max_length=MAX_FIELD_NAME_CHARS)]] = Field(
        default_factory=list, max_length=MAX_FIELDS
    )
    key_field: str | None = Field(default=None, max_length=MAX_FIELD_NAME_CHARS)
    clear_key: bool = False


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
