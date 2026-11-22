"""The collection store. Every function takes the owner, and a collection or
row that is not theirs answers as if it did not exist.

Each write runs in one transaction holding the collection's row lock, so
parallel writers to one collection take turns: two upserts of one key make one
row, and parallel adds cannot pass the row limit together.
"""

from typing import Any

from prisma.models import AgentGraph, Collection, CollectionRow

from backend.data.db import query_raw_with_schema, transaction
from backend.util.json import SafeJson

from .fields import FieldSet, check_field_name, check_row_size, coerce, key_text
from .merge import RowState, accept_pending, automated_write, new_row, person_edit
from .model import (
    MAX_COLLECTIONS_PER_USER,
    MAX_NAME_CHARS,
    MAX_ROWS,
    OWNER,
    Aggregate,
    CollectionError,
    CollectionField,
    CollectionInfo,
    CollectionNotFoundError,
    CollectionRowRecord,
    Condition,
    FieldSpec,
    RowNotFoundError,
    RowPage,
    RowQuery,
    StaleRowError,
    Writer,
    WriteResult,
)
from .query import Params, aggregate_expr, order_clause, where_clause, where_conditions
from .schema import SchemaChange, apply_schema_change


async def list_collections(user_id: str) -> list[CollectionInfo]:
    rows = await Collection.prisma().find_many(
        where={"userId": user_id},
        order={"updatedAt": "desc"},
        take=MAX_COLLECTIONS_PER_USER,
    )
    counts = await _row_counts(user_id, [row.id for row in rows])
    return [_info(row, counts.get(row.id, 0)) for row in rows]


async def get_collection(user_id: str, ref: str) -> CollectionInfo | None:
    """The user's collection by id, or by name without case."""
    row = await _find(user_id, ref)
    if row is None:
        return None
    counts = await _row_counts(user_id, [row.id])
    return _info(row, counts.get(row.id, 0))


async def create_collection(
    user_id: str,
    name: str,
    fields: list[FieldSpec],
    key_field: str | None = None,
) -> CollectionInfo:
    async with transaction() as tx:
        row = await _create(tx, user_id, name, fields, key_field)
    return _info(row, 0)


async def write_rows(
    user_id: str,
    collection: str,
    rows: list[dict[str, Any]],
    *,
    writer: Writer,
    match_on_key: bool = False,
    write_id: str | None = None,
    create: bool = True,
) -> WriteResult:
    """Add *rows*, or with *match_on_key* update the row each key names and
    add the rest. A field no row had before is added with an inferred type;
    *write_id* makes a retried add of a keyless row a no-op."""
    async with transaction() as tx:
        locked = await _lock(tx, user_id, collection, create=create)
        fields = FieldSet(_fields(locked))
        key_field = _key_field(fields, locked.keyField)
        if match_on_key and key_field is None:
            raise CollectionError(
                f"'{locked.name}' has no key field to match on; give it one, or add "
                "without matching"
            )
        count = await CollectionRow.prisma(tx).count(where={"collectionId": locked.id})
        result = WriteResult(collection=_info(locked, count), rows=[])
        for index, raw in enumerate(rows):
            values = fields.coerce_row(raw, grow=True)
            key = key_text(values.get(key_field)) if key_field else None
            step = f"{write_id}#{index}" if write_id and len(rows) > 1 else write_id
            existing = await _existing(tx, locked.id, key, None if key else step)
            if existing is not None and key is None:
                result.rows.append(_record(existing))
                continue
            if existing is not None:
                if not match_on_key:
                    raise CollectionError(
                        f"A row with {key_field} '{key}' exists; match on key to update it"
                    )
                merged = automated_write(_state(existing), values)
                for name in merged.held:
                    result.held[name] = result.held.get(name, 0) + 1
                saved = (
                    await _save(tx, existing, merged.state, writer, key=key)
                    if merged.changed
                    else existing
                )
                result.updated += int(merged.changed)
                result.rows.append(_record(saved))
                continue
            if count >= MAX_ROWS:
                raise CollectionError(
                    f"'{locked.name}' holds the most rows a collection takes ({MAX_ROWS:,})"
                )
            created = await _insert(
                tx, locked, new_row(values, by_person=False), writer, key, step
            )
            count += 1
            result.added += 1
            result.rows.append(_record(created))
        locked = await _store_fields(tx, locked, fields)
    result.collection = _info(locked, count)
    return result


async def update_row(
    user_id: str,
    collection: str,
    values: dict[str, Any],
    *,
    writer: Writer,
    key: str | None = None,
    row_id: str | None = None,
) -> CollectionRowRecord:
    """An automated change to the named fields of one row, found by key or id."""
    async with transaction() as tx:
        locked = await _lock(tx, user_id, collection, create=False)
        fields = FieldSet(_fields(locked))
        row = await _one(tx, locked, key=key, row_id=row_id)
        coerced = fields.coerce_row(values, grow=True)
        merged = automated_write(_state(row), coerced)
        key_field = _key_field(fields, locked.keyField)
        saved = row
        if merged.changed:
            new_key = key_text(merged.state.data.get(key_field)) if key_field else None
            await _refuse_duplicate_key(tx, locked, new_key, row.id)
            saved = await _save(tx, row, merged.state, writer, key=new_key)
        await _store_fields(tx, locked, fields)
    return _record(saved)


async def get_row(
    user_id: str, collection: str, *, key: str | None = None, row_id: str | None = None
) -> CollectionRowRecord | None:
    found = await _find(user_id, collection)
    if found is None:
        raise CollectionNotFoundError(f"No collection named '{collection}'")
    row = await CollectionRow.prisma().find_first(
        where=_row_where(user_id, found.id, key=_lookup_key(found, key), row_id=row_id)
    )
    return _record(row) if row else None


async def delete_rows(
    user_id: str,
    collection: str,
    *,
    keys: list[str] | None = None,
    row_ids: list[str] | None = None,
) -> int:
    found = await _find(user_id, collection)
    if found is None:
        raise CollectionNotFoundError(f"No collection named '{collection}'")
    match: list[dict[str, Any]] = []
    if keys:
        match.append({"key": {"in": [_lookup_key(found, k) for k in keys]}})
    if row_ids:
        match.append({"id": {"in": row_ids}})
    if not match:
        return 0
    return await CollectionRow.prisma().delete_many(
        where={"userId": user_id, "collectionId": found.id, "OR": match}
    )


async def query_rows(user_id: str, collection: str, query: RowQuery) -> RowPage:
    found = await _find(user_id, collection)
    if found is None:
        raise CollectionNotFoundError(f"No collection named '{collection}'")
    fields = FieldSet(_fields(found))
    params = Params(user_id, found.id)
    where = where_clause(fields, query, params)
    order = order_clause(fields, query.sort, params)
    limit, offset = params.add(query.limit), params.add(query.offset)
    ids = await query_raw_with_schema(
        f'SELECT r."id" FROM {{schema_prefix}}"CollectionRow" r WHERE {where} '
        f"ORDER BY {order} LIMIT {limit} OFFSET {offset}",
        *params.values,
    )
    total = await _count_where(fields, query, user_id, found.id)
    ordered = [row["id"] for row in ids]
    rows = (
        await CollectionRow.prisma().find_many(
            where={"id": {"in": ordered}, "userId": user_id}
        )
        if ordered
        else []
    )
    by_id = {row.id: row for row in rows}
    return RowPage(rows=[_record(by_id[i]) for i in ordered if i in by_id], total=total)


async def aggregate_rows(
    user_id: str,
    collection: str,
    aggregates: list[Aggregate],
    conditions: list[Condition] | None = None,
) -> list[float | None]:
    """Each aggregate over the rows that meet *conditions*, in one query."""
    found = await _find(user_id, collection)
    if found is None:
        raise CollectionNotFoundError(f"No collection named '{collection}'")
    if not aggregates:
        return []
    fields = FieldSet(_fields(found))
    params = Params(user_id, found.id)
    where = where_conditions(fields, conditions or [], params)
    columns = ", ".join(
        f"{aggregate_expr(fields, a, params)} AS a{i}" for i, a in enumerate(aggregates)
    )
    (row,) = await query_raw_with_schema(
        f'SELECT {columns} FROM {{schema_prefix}}"CollectionRow" r WHERE {where}',
        *params.values,
    )
    return [_float(row[f"a{i}"]) for i in range(len(aggregates))]


async def add_row(
    user_id: str, collection_id: str, values: dict[str, Any]
) -> CollectionRowRecord:
    """A row the owner adds by hand: every field they fill is theirs."""
    async with transaction() as tx:
        locked = await _lock(tx, user_id, collection_id, create=False)
        fields = FieldSet(_fields(locked))
        coerced = fields.coerce_row(values, grow=False)
        key_field = _key_field(fields, locked.keyField)
        key = key_text(coerced.get(key_field)) if key_field else None
        await _refuse_duplicate_key(tx, locked, key, None)
        count = await CollectionRow.prisma(tx).count(where={"collectionId": locked.id})
        if count >= MAX_ROWS:
            raise CollectionError(
                f"'{locked.name}' holds the most rows a collection takes ({MAX_ROWS:,})"
            )
        created = await _insert(
            tx, locked, new_row(coerced, by_person=True), OWNER, key, None
        )
        await _store_fields(tx, locked, fields)
    return _record(created)


async def edit_row(
    user_id: str,
    collection_id: str,
    row_id: str,
    values: dict[str, Any],
    *,
    based_on: int,
) -> CollectionRowRecord:
    """The owner's edit, refused when the row changed after the version they saw."""
    async with transaction() as tx:
        locked = await _lock(tx, user_id, collection_id, create=False)
        fields = FieldSet(_fields(locked))
        row = await _one(tx, locked, row_id=row_id)
        if row.revision != based_on:
            raise StaleRowError(
                "This row changed since you opened it; it shows the new values now",
                _record(row),
            )
        merged = person_edit(_state(row), fields.coerce_row(values, grow=False))
        key_field = _key_field(fields, locked.keyField)
        key = key_text(merged.state.data.get(key_field)) if key_field else None
        await _refuse_duplicate_key(tx, locked, key, row.id)
        saved = await _save(tx, row, merged.state, OWNER, key=key)
        await _store_fields(tx, locked, fields)
    return _record(saved)


async def accept_pending_values(
    user_id: str, collection_id: str, row_id: str, names: list[str], *, based_on: int
) -> CollectionRowRecord:
    """Apply the newer automated values held back on *names*."""
    async with transaction() as tx:
        locked = await _lock(tx, user_id, collection_id, create=False)
        row = await _one(tx, locked, row_id=row_id)
        if row.revision != based_on:
            raise StaleRowError(
                "This row changed since you opened it; it shows the new values now",
                _record(row),
            )
        merged = accept_pending(_state(row), names)
        if not merged.changed:
            return _record(row)
        key_field = _key_field(FieldSet(_fields(locked)), locked.keyField)
        key = key_text(merged.state.data.get(key_field)) if key_field else None
        await _refuse_duplicate_key(tx, locked, key, row.id)
        saved = await _save(tx, row, merged.state, OWNER, key=key)
    return _record(saved)


async def delete_collection(user_id: str, collection: str) -> bool:
    found = await _find(user_id, collection)
    if found is None:
        return False
    deleted = await Collection.prisma().delete_many(
        where={"id": found.id, "userId": user_id}
    )
    return deleted > 0


async def change_schema(
    user_id: str, collection: str, change: SchemaChange
) -> CollectionInfo:
    """The owner's change to a collection's name, fields or key."""
    async with transaction() as tx:
        locked = await _lock(tx, user_id, collection, create=False)
        if change.rename and change.rename.strip().casefold() != locked.nameKey:
            await _refuse_taken_name(tx, user_id, change.rename)
        updated = await apply_schema_change(tx, locked, change)
        count = await CollectionRow.prisma(tx).count(where={"collectionId": locked.id})
    return _info(updated, count)


async def graph_created_by(user_id: str, graph_id: str) -> bool:
    """Whether *user_id* made the workflow, rather than installing someone else's."""
    return (
        await AgentGraph.prisma().count(where={"id": graph_id, "userId": user_id})
    ) > 0


async def _find(user_id: str, ref: str) -> Collection | None:
    by_id = await Collection.prisma().find_first(where={"id": ref, "userId": user_id})
    if by_id is not None:
        return by_id
    return await Collection.prisma().find_first(
        where={"userId": user_id, "nameKey": ref.strip().casefold()}
    )


async def _lock(tx: Any, user_id: str, ref: str, *, create: bool) -> Collection:
    """The collection, its row locked for this transaction; with *create*, made
    first when the user has none by that name."""
    found = await query_raw_with_schema(
        'SELECT "id" FROM {schema_prefix}"Collection" '
        'WHERE "userId" = $1 AND ("id" = $2 OR "nameKey" = $3) '
        'ORDER BY ("id" = $2) DESC LIMIT 1 FOR UPDATE',
        user_id,
        ref,
        ref.strip().casefold(),
        client=tx,
    )
    if not found:
        if not create:
            raise CollectionNotFoundError(f"No collection named '{ref}'")
        created = await _create(tx, user_id, ref, [], None)
        return await _lock(tx, user_id, created.id, create=False)
    row = await Collection.prisma(tx).find_unique(where={"id": found[0]["id"]})
    assert row is not None
    return row


async def _create(
    tx: Any,
    user_id: str,
    name: str,
    specs: list[FieldSpec],
    key_field: str | None,
) -> Collection:
    clean = " ".join(name.split())
    if not clean or len(clean) > MAX_NAME_CHARS:
        raise CollectionError(f"A collection name is 1 to {MAX_NAME_CHARS} characters")
    # Parallel first writes take turns on the owner, so the cap and the name hold.
    await query_raw_with_schema(
        'SELECT "id" FROM {schema_prefix}"User" WHERE "id" = $1 FOR UPDATE',
        user_id,
        client=tx,
    )
    await _refuse_taken_name(tx, user_id, clean)
    if await Collection.prisma(tx).count(where={"userId": user_id}) >= (
        MAX_COLLECTIONS_PER_USER
    ):
        raise CollectionError(
            "You have the most collections one account can keep "
            f"({MAX_COLLECTIONS_PER_USER})"
        )
    fields = FieldSet([])
    for spec in specs:
        check_field_name(spec.name)
        fields.add(CollectionField(**spec.model_dump()))
    key = fields.require(key_field).name if key_field else None
    return await Collection.prisma(tx).create(
        data={
            "userId": user_id,
            "name": clean,
            "nameKey": clean.casefold(),
            "fields": SafeJson([f.model_dump(mode="json") for f in fields.fields]),
            "keyField": key,
        }
    )


async def _refuse_taken_name(tx: Any, user_id: str, name: str) -> None:
    clean = " ".join(name.split())
    taken = await Collection.prisma(tx).find_first(
        where={"userId": user_id, "nameKey": clean.casefold()}
    )
    if taken is not None:
        raise CollectionError(f"You already have a collection named '{taken.name}'")


async def _existing(
    tx: Any, collection_id: str, key: str | None, write_id: str | None
) -> CollectionRow | None:
    if key is not None:
        return await CollectionRow.prisma(tx).find_first(
            where={"collectionId": collection_id, "key": key}
        )
    if write_id is not None:
        return await CollectionRow.prisma(tx).find_first(
            where={"collectionId": collection_id, "writeId": write_id}
        )
    return None


async def _one(
    tx: Any, locked: Collection, *, key: str | None = None, row_id: str | None = None
) -> CollectionRow:
    row = await CollectionRow.prisma(tx).find_first(
        where=_row_where(
            locked.userId, locked.id, key=_lookup_key(locked, key), row_id=row_id
        )
    )
    if row is None:
        what = f"key '{key}'" if key is not None else f"id '{row_id}'"
        raise RowNotFoundError(f"'{locked.name}' has no row with {what}")
    return row


def _row_where(
    user_id: str, collection_id: str, *, key: str | None, row_id: str | None
) -> Any:
    if key is None and row_id is None:
        raise CollectionError("Name the row by its key or its id")
    where: dict[str, Any] = {"userId": user_id, "collectionId": collection_id}
    if row_id is not None:
        where["id"] = row_id
    else:
        where["key"] = key
    return where


async def _refuse_duplicate_key(
    tx: Any, locked: Collection, key: str | None, row_id: str | None
) -> None:
    if key is None:
        return
    clash = await CollectionRow.prisma(tx).find_first(
        where={"collectionId": locked.id, "key": key}
    )
    if clash is not None and clash.id != row_id:
        raise CollectionError(
            f"Another row in '{locked.name}' already has {locked.keyField} '{key}'"
        )


async def _insert(
    tx: Any,
    locked: Collection,
    state: RowState,
    writer: Writer,
    key: str | None,
    write_id: str | None,
) -> CollectionRow:
    check_row_size(state.data)
    return await CollectionRow.prisma(tx).create(
        data={
            "collectionId": locked.id,
            "userId": locked.userId,
            "key": key,
            "writeId": write_id,
            "data": SafeJson(state.data),
            "sourceData": SafeJson(state.source),
            "pending": SafeJson(state.pending),
            "manualFields": state.manual,
            "writtenBy": writer,
        }
    )


async def _save(
    tx: Any, row: CollectionRow, state: RowState, writer: Writer, *, key: str | None
) -> CollectionRow:
    check_row_size(state.data)
    saved = await CollectionRow.prisma(tx).update(
        where={"id": row.id},
        data={
            "key": key,
            "data": SafeJson(state.data),
            "sourceData": SafeJson(state.source),
            "pending": SafeJson(state.pending),
            "manualFields": state.manual,
            "writtenBy": writer,
            "revision": {"increment": 1},
        },
    )
    assert saved is not None
    return saved


async def _store_fields(tx: Any, locked: Collection, fields: FieldSet) -> Collection:
    if not fields.changed:
        return locked
    updated = await Collection.prisma(tx).update(
        where={"id": locked.id},
        data={
            "fields": SafeJson([f.model_dump(mode="json") for f in fields.fields]),
            "revision": {"increment": 1},
        },
    )
    assert updated is not None
    return updated


async def _count_where(
    fields: FieldSet, query: RowQuery, user_id: str, collection_id: str
) -> int:
    params = Params(user_id, collection_id)
    where = where_clause(fields, query, params)
    (row,) = await query_raw_with_schema(
        f'SELECT count(*)::int AS n FROM {{schema_prefix}}"CollectionRow" r WHERE {where}',
        *params.values,
    )
    return int(row["n"])


async def _row_counts(user_id: str, ids: list[str]) -> dict[str, int]:
    if not ids:
        return {}
    grouped = await CollectionRow.prisma().group_by(
        by=["collectionId"],
        where={"userId": user_id, "collectionId": {"in": ids}},
        count=True,
    )
    return {g["collectionId"]: int(g["_count"]["_all"]) for g in grouped}


def _lookup_key(row: Collection, key: str | None) -> str | None:
    """*key* in the form rows store it, read as the key field's type."""
    if key is None:
        return None
    fields = FieldSet(_fields(row))
    key_field = _key_field(fields, row.keyField)
    if key_field is None:
        raise CollectionError(f"'{row.name}' has no key field; name the row by its id")
    try:
        return key_text(coerce(fields.require(key_field), key))
    except (TypeError, ValueError):
        return key


def _key_field(fields: FieldSet, key_field: str | None) -> str | None:
    if key_field is None:
        return None
    found = fields.find(key_field)
    return found.name if found else None


def _fields(row: Collection) -> list[CollectionField]:
    stored = row.fields if isinstance(row.fields, list) else []
    return [CollectionField.model_validate(f) for f in stored]


def _state(row: CollectionRow) -> RowState:
    return RowState(
        data=_as_dict(row.data),
        source=_as_dict(row.sourceData),
        pending=_as_dict(row.pending),
        manual=list(row.manualFields or []),
    )


def _info(row: Collection, count: int) -> CollectionInfo:
    return CollectionInfo(
        id=row.id,
        name=row.name,
        fields=_fields(row),
        key_field=row.keyField,
        revision=row.revision,
        row_count=count,
        updated_at=row.updatedAt,
    )


def _record(row: CollectionRow) -> CollectionRowRecord:
    return CollectionRowRecord(
        id=row.id,
        key=row.key,
        values=_as_dict(row.data),
        pending=_as_dict(row.pending),
        manual_fields=list(row.manualFields or []),
        written_by=row.writtenBy,
        revision=row.revision,
        created_at=row.createdAt,
        updated_at=row.updatedAt,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    return None if value is None else float(value)
