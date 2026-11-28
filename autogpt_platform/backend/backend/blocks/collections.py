"""Blocks that read and write the user's collections: tables of their own,
named, typed, and shown on their pages.

They run only in workflows the user made: an installed workflow could name a
collection its author never saw.
"""

from typing import Any, Literal

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.data.collections.model import (
    MAX_CONDITIONS,
    MAX_QUERY_LIMIT,
    CollectionError,
    CollectionInfo,
    CollectionRowRecord,
    Condition,
    FieldSpec,
    RowPage,
    RowQuery,
    Sort,
    WriteResult,
    workflow_writer,
)
from backend.data.db_accessors import collections_db
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.feature_flag import Flag, is_feature_enabled

INSTALLED_REFUSAL = (
    "Collection blocks run only in workflows you made. This one was installed, "
    "so it can't read or write your collections."
)
_COLLECTION_HELP = "The collection's name; one is made on the first write."


class CollectionBlockBase(Block):
    async def _guard(self, user_id: str, graph_id: str) -> None:
        if not await is_feature_enabled(Flag.COLLECTIONS, user_id, default=False):
            raise CollectionError("Collections are not available on your account yet")
        if graph_id.startswith(COPILOT_SESSION_PREFIX):
            return
        if not await collections_db().graph_created_by(user_id, graph_id):
            raise CollectionError(INSTALLED_REFUSAL)


class AddToCollectionBlock(CollectionBlockBase):
    """Adds a row to a collection, or with "match on key" updates the row with
    that key and adds it otherwise. A field the collection lacks is added."""

    class Input(BlockSchemaInput):
        collection: str = SchemaField(description=_COLLECTION_HELP)
        row: dict[str, Any] = SchemaField(description="Field name to value.")
        match_on_key: bool = SchemaField(
            description="Update the row whose key field holds this row's key.",
            default=False,
        )

    class Output(BlockSchemaOutput):
        row: dict[str, Any] = SchemaField(description="The row as stored, with _id.")
        row_id: str = SchemaField(description="The row's id.")
        added: bool = SchemaField(
            description="True for a new row, False for an update."
        )
        kept_fields: list[str] = SchemaField(
            description="Fields where a person's edit was kept over this value."
        )

    def __init__(self):
        super().__init__(
            id="5c2f0b7e-4a6d-4e35-9d0e-6a3f1b2c8d41",
            description=_doc(AddToCollectionBlock),
            categories={BlockCategory.DATA},
            input_schema=AddToCollectionBlock.Input,
            output_schema=AddToCollectionBlock.Output,
            test_input={
                "collection": "Content ideas",
                "row": {"keyword": "cohort retention chart", "volume": 1900},
                "match_on_key": True,
            },
            test_output=[
                (
                    "row",
                    {"keyword": "cohort retention chart", "volume": 1900, "_id": "r1"},
                ),
                ("row_id", "r1"),
                ("added", True),
                ("kept_fields", []),
            ],
            test_mock={
                "_guard": lambda *args, **kwargs: None,
                "_write": lambda *args, **kwargs: _sample_write(),
            },
            effect=BlockEffect.WORKSPACE,
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        graph_id: str,
        node_exec_id: str,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        await self._guard(user_id, graph_id)
        if execution_context.dry_run:
            yield "row", {**input_data.row, "_id": ""}
            return
        result = await self._write(
            user_id,
            input_data.collection,
            input_data.row,
            graph_id=graph_id,
            match_on_key=input_data.match_on_key,
            write_id=node_exec_id,
        )
        (row,) = result.rows
        yield "row", _row(row)
        yield "row_id", row.id
        yield "added", result.added == 1
        yield "kept_fields", sorted(result.held)

    async def _write(
        self,
        user_id: str,
        collection: str,
        row: dict[str, Any],
        *,
        graph_id: str,
        match_on_key: bool,
        write_id: str,
    ) -> WriteResult:
        return await collections_db().write_rows(
            user_id,
            collection,
            [row],
            writer=workflow_writer(graph_id),
            match_on_key=match_on_key,
            write_id=write_id,
        )


class UpdateCollectionRowBlock(CollectionBlockBase):
    """Changes the named fields of one row in a collection, found by its key or id."""

    class Input(BlockSchemaInput):
        collection: str = SchemaField(description="The collection's name.")
        key: str | None = SchemaField(
            description="The row's value in the key field.", default=None
        )
        row_id: str | None = SchemaField(description="Or the row's id.", default=None)
        values: dict[str, Any] = SchemaField(description="Field name to new value.")

    class Output(BlockSchemaOutput):
        row: dict[str, Any] = SchemaField(description="The row as stored, with _id.")
        row_id: str = SchemaField(description="The row's id.")

    def __init__(self):
        super().__init__(
            id="8e1d4c3a-2b7f-4f60-a5d9-3c0e7b1f2a96",
            description=_doc(UpdateCollectionRowBlock),
            categories={BlockCategory.DATA},
            input_schema=UpdateCollectionRowBlock.Input,
            output_schema=UpdateCollectionRowBlock.Output,
            test_input={
                "collection": "Content ideas",
                "key": "cohort retention chart",
                "values": {"status": "Drafted"},
            },
            test_output=[
                (
                    "row",
                    {"keyword": "cohort retention chart", "volume": 1900, "_id": "r1"},
                ),
                ("row_id", "r1"),
            ],
            test_mock={
                "_guard": lambda *args, **kwargs: None,
                "_update": lambda *args, **kwargs: _sample_row(),
            },
            effect=BlockEffect.WORKSPACE,
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        graph_id: str,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        await self._guard(user_id, graph_id)
        if execution_context.dry_run:
            yield "row", {**input_data.values, "_id": input_data.row_id or ""}
            return
        row = await self._update(user_id, input_data, graph_id=graph_id)
        yield "row", _row(row)
        yield "row_id", row.id

    async def _update(
        self, user_id: str, input_data: Input, *, graph_id: str
    ) -> CollectionRowRecord:
        return await collections_db().update_row(
            user_id,
            input_data.collection,
            input_data.values,
            writer=workflow_writer(graph_id),
            key=input_data.key,
            row_id=input_data.row_id,
        )


class FindCollectionRowsBlock(CollectionBlockBase):
    """Finds rows in a collection that meet up to three conditions, sorted,
    up to a limit; gives the rows as a list, each row on its own, and a count."""

    class Input(BlockSchemaInput):
        collection: str = SchemaField(description="The collection's name.")
        conditions: list[Condition] = SchemaField(
            description="All must hold: is, is_not, contains, gt, lt, gte, lte, "
            "empty or not_empty.",
            default_factory=list,
            max_length=MAX_CONDITIONS,
        )
        sort_by: str | None = SchemaField(
            description="A field to sort on.", default=None
        )
        sort_direction: Literal["asc", "desc"] = SchemaField(
            description="Smallest first, or largest first.", default="asc"
        )
        limit: int = SchemaField(
            description="At most this many rows.", default=50, ge=1, le=MAX_QUERY_LIMIT
        )

    class Output(BlockSchemaOutput):
        rows: list[dict[str, Any]] = SchemaField(description="The rows, with _id.")
        row: dict[str, Any] = SchemaField(description="Each row, one at a time.")
        count: int = SchemaField(description="How many rows meet the conditions.")

    def __init__(self):
        super().__init__(
            id="2a9b6e1f-7c3d-4b8a-9f05-d4e2c6a1b37e",
            description=_doc(FindCollectionRowsBlock),
            categories={BlockCategory.DATA},
            input_schema=FindCollectionRowsBlock.Input,
            output_schema=FindCollectionRowsBlock.Output,
            test_input={
                "collection": "Follow-ups",
                "conditions": [{"field": "done", "op": "is", "value": False}],
                "sort_by": "due",
            },
            test_output=[
                (
                    "rows",
                    [
                        {
                            "keyword": "cohort retention chart",
                            "volume": 1900,
                            "_id": "r1",
                        }
                    ],
                ),
                (
                    "row",
                    {"keyword": "cohort retention chart", "volume": 1900, "_id": "r1"},
                ),
                ("count", 1),
            ],
            test_mock={
                "_guard": lambda *args, **kwargs: None,
                "_find": lambda *args, **kwargs: RowPage(rows=[_sample_row()], total=1),
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, user_id: str, graph_id: str, **kwargs
    ) -> BlockOutput:
        await self._guard(user_id, graph_id)
        query = RowQuery(
            conditions=input_data.conditions,
            sort=(
                Sort(field=input_data.sort_by, direction=input_data.sort_direction)
                if input_data.sort_by
                else None
            ),
            limit=input_data.limit,
        )
        page = await self._find(user_id, input_data.collection, query)
        rows = [_row(row) for row in page.rows]
        yield "rows", rows
        for row in rows:
            yield "row", row
        yield "count", page.total

    async def _find(self, user_id: str, collection: str, query: RowQuery) -> RowPage:
        return await collections_db().query_rows(user_id, collection, query)


class GetCollectionRowBlock(CollectionBlockBase):
    """Reads one row of a collection by its key or id."""

    class Input(BlockSchemaInput):
        collection: str = SchemaField(description="The collection's name.")
        key: str | None = SchemaField(
            description="The row's value in the key field.", default=None
        )
        row_id: str | None = SchemaField(description="Or the row's id.", default=None)

    class Output(BlockSchemaOutput):
        row: dict[str, Any] = SchemaField(description="The row, with _id.")
        not_found: str = SchemaField(description="Set when no row matches.")

    def __init__(self):
        super().__init__(
            id="b47c1e9d-0f2a-4d6b-8e3c-7a5f9d1b2c60",
            description=_doc(GetCollectionRowBlock),
            categories={BlockCategory.DATA},
            input_schema=GetCollectionRowBlock.Input,
            output_schema=GetCollectionRowBlock.Output,
            test_input={"collection": "Content ideas", "key": "cohort retention chart"},
            test_output=[
                (
                    "row",
                    {"keyword": "cohort retention chart", "volume": 1900, "_id": "r1"},
                )
            ],
            test_mock={
                "_guard": lambda *args, **kwargs: None,
                "_get": lambda *args, **kwargs: _sample_row(),
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, user_id: str, graph_id: str, **kwargs
    ) -> BlockOutput:
        await self._guard(user_id, graph_id)
        row = await self._get(user_id, input_data)
        if row is None:
            yield "not_found", f"No row in '{input_data.collection}' matches"
            return
        yield "row", _row(row)

    async def _get(self, user_id: str, input_data: Input) -> CollectionRowRecord | None:
        return await collections_db().get_row(
            user_id, input_data.collection, key=input_data.key, row_id=input_data.row_id
        )


class DeleteCollectionRowBlock(CollectionBlockBase):
    """Deletes one row of a collection by its key or id."""

    class Input(BlockSchemaInput):
        collection: str = SchemaField(description="The collection's name.")
        key: str | None = SchemaField(
            description="The row's value in the key field.", default=None
        )
        row_id: str | None = SchemaField(description="Or the row's id.", default=None)

    class Output(BlockSchemaOutput):
        deleted: bool = SchemaField(description="False when no row matched.")

    def __init__(self):
        super().__init__(
            id="d93e5a27-6b1c-4f8e-a0d4-1e7c3b9f5a82",
            description=_doc(DeleteCollectionRowBlock),
            categories={BlockCategory.DATA},
            input_schema=DeleteCollectionRowBlock.Input,
            output_schema=DeleteCollectionRowBlock.Output,
            test_input={"collection": "Stock", "key": "TT-104"},
            test_output=[("deleted", True)],
            test_mock={
                "_guard": lambda *args, **kwargs: None,
                "_delete": lambda *args, **kwargs: 1,
            },
            effect=BlockEffect.WORKSPACE,
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        graph_id: str,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        await self._guard(user_id, graph_id)
        if not (input_data.key or input_data.row_id):
            raise CollectionError("Name the row by its key or its id")
        if execution_context.dry_run:
            yield "deleted", False
            return
        yield "deleted", await self._delete(user_id, input_data) > 0

    async def _delete(self, user_id: str, input_data: Input) -> int:
        return await collections_db().delete_rows(
            user_id,
            input_data.collection,
            keys=[input_data.key] if input_data.key else None,
            row_ids=[input_data.row_id] if input_data.row_id else None,
        )


class CreateCollectionBlock(CollectionBlockBase):
    """Declares a collection's fields and key up front, for when types inferred
    from the first write are not wanted. An existing collection is left as it is."""

    class Input(BlockSchemaInput):
        name: str = SchemaField(description="The collection's name.")
        fields: list[FieldSpec] = SchemaField(
            description="Each field's name and type: text, number, date, select, "
            "multi_select, checkbox, url or email.",
        )
        key_field: str | None = SchemaField(
            description="A field whose value identifies one row.", default=None
        )

    class Output(BlockSchemaOutput):
        collection_id: str = SchemaField(description="The collection's id.")
        created: bool = SchemaField(description="False when it already existed.")

    def __init__(self):
        super().__init__(
            id="7f6a2d1c-9e4b-4c37-b8a1-5d0f3e6c9b24",
            description=_doc(CreateCollectionBlock),
            categories={BlockCategory.DATA},
            input_schema=CreateCollectionBlock.Input,
            output_schema=CreateCollectionBlock.Output,
            test_input={
                "name": "Stock",
                "fields": [
                    {"name": "sku", "type": "text"},
                    {"name": "on_hand", "type": "number"},
                ],
                "key_field": "sku",
            },
            test_output=[("collection_id", "c1"), ("created", True)],
            test_mock={
                "_guard": lambda *args, **kwargs: None,
                "_existing": lambda *args, **kwargs: None,
                "_create": lambda *args, **kwargs: _sample_write().collection,
            },
            effect=BlockEffect.WORKSPACE,
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        graph_id: str,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        await self._guard(user_id, graph_id)
        existing = await self._existing(user_id, input_data.name)
        if existing is not None:
            yield "collection_id", existing.id
            yield "created", False
            return
        if execution_context.dry_run:
            yield "created", False
            return
        created = await self._create(user_id, input_data)
        yield "collection_id", created.id
        yield "created", True

    async def _existing(self, user_id: str, name: str) -> CollectionInfo | None:
        return await collections_db().get_collection(user_id, name)

    async def _create(self, user_id: str, input_data: Input) -> CollectionInfo:
        return await collections_db().create_collection(
            user_id, input_data.name, input_data.fields, input_data.key_field
        )


def _row(row: CollectionRowRecord) -> dict[str, Any]:
    return {**row.values, "_id": row.id}


def _doc(block: type) -> str:
    return " ".join((block.__doc__ or "").split())


def _sample_row() -> CollectionRowRecord:
    return CollectionRowRecord.model_validate(
        {
            "id": "r1",
            "key": "cohort retention chart",
            "values": {"keyword": "cohort retention chart", "volume": 1900},
            "written_by": "workflow:g1",
            "revision": 1,
            "created_at": "2026-11-22T09:00:00Z",
            "updated_at": "2026-11-22T09:00:00Z",
        }
    )


def _sample_write() -> WriteResult:
    collection = CollectionInfo.model_validate(
        {
            "id": "c1",
            "name": "Content ideas",
            "fields": [],
            "revision": 1,
            "updated_at": "2026-11-22T09:00:00Z",
        }
    )
    return WriteResult(collection=collection, rows=[_sample_row()], added=1)
