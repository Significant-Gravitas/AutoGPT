"""CollectionsTool - Otto and experts read and write the user's collections."""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.copilot.gate.policy import Effect
from backend.copilot.gate.subject import Subject
from backend.copilot.model import ChatSession
from backend.data.collections.model import (
    MAX_COLLECTIONS_PER_USER,
    MAX_CONDITIONS,
    MAX_FIELDS,
    MAX_ROW_BYTES,
    MAX_ROWS,
    OTTO,
    CollectionError,
    CollectionInfo,
    CollectionRowRecord,
    Condition,
    FieldSpec,
    FieldType,
    RowQuery,
    Sort,
    expert_writer,
)
from backend.data.collections.schema import SchemaChange
from backend.data.db_accessors import collections_db
from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

COLLECTIONS_TOOL = "collections"
MAX_ROWS_PER_CALL = 50
_READS = frozenset({None, "list", "describe", "find"})
_ROW_WRITES = frozenset({"create", "add", "update"})
# Deletes and shape changes go through the approval gate as platform edits.
_OWNER_CHANGES = frozenset({"delete_rows", "change_schema", "delete_collection"})
ACTIONS = sorted(a for a in _READS | _ROW_WRITES | _OWNER_CHANGES if a is not None)


class CollectionsResponse(ToolResponseBase):
    """A collection's shape, rows from it, or the catalogue of actions."""

    type: ResponseType = ResponseType.COLLECTIONS
    collections: list[CollectionInfo] | None = None
    collection: CollectionInfo | None = None
    rows: list[CollectionRowRecord] | None = None
    total: int | None = None
    catalogue: dict[str, Any] | None = None


class _Find(BaseModel):
    conditions: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    sort: Sort | None = None
    limit: int = Field(default=20, ge=1, le=MAX_ROWS_PER_CALL)


class CollectionsTool(BaseTool):
    has_gate_subject = True

    @property
    def name(self) -> str:
        return COLLECTIONS_TOOL

    @property
    def description(self) -> str:
        return (
            "Read and write the user's collections: tables of their own that "
            "workflows and pages share. Keep a list here when the user tells you "
            "something to track (follow-ups, candidates, ideas). Call with no "
            "arguments for the actions and field types."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ACTIONS,
                    "description": "Omit for the catalogue.",
                },
                "collection": {
                    "type": "string",
                    "description": "A collection's name or id.",
                    "entity": None,
                },
                "rows": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "add: up to 50 rows, field name to value.",
                },
                "match_on_key": {
                    "type": "boolean",
                    "description": "add: update the row with the same key.",
                },
                "key": {"type": "string", "description": "update: the row's key."},
                "row_id": {
                    "type": "string",
                    "description": "update: or its id.",
                    "entity": None,
                },
                "values": {
                    "type": "object",
                    "description": "update: field name to new value.",
                },
                "find": {
                    "type": "object",
                    "description": "find: conditions, sort, limit (see catalogue).",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "create: name and type of each field.",
                },
                "key_field": {
                    "type": "string",
                    "description": "create: the field whose value identifies a row.",
                },
                "change": {
                    "type": "object",
                    "description": "change_schema: rename, add_fields, "
                    "remove_fields, key_field.",
                },
                "row_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "delete_rows: the rows' ids.",
                    "entity": None,
                },
            },
        }

    async def gate_subject(
        self, user_id: str, session: ChatSession, args: dict[str, Any]
    ) -> Subject | None:
        """A read asks nothing and a row write is a workspace write; deletes and
        schema changes fall to the tool's platform effect."""
        action = args.get("action")
        name = str(args.get("collection") or "collections")
        if action in _READS:
            return Subject(key=COLLECTIONS_TOOL, name=name, effect=Effect.READ)
        if action in _ROW_WRITES:
            return Subject(key=COLLECTIONS_TOOL, name=name, effect=Effect.WORKSPACE)
        return None

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id or not await is_feature_enabled(Flag.COLLECTIONS, user_id):
            return _error("Collections are not available.", "unavailable", session_id)
        action = kwargs.get("action")
        if action is None:
            return CollectionsResponse(
                message="The actions and field types. Call again with an action.",
                session_id=session_id,
                catalogue=catalogue(),
            ).from_outside()
        run = _ACTIONS.get(action)
        if run is None:
            return _error(f"No action '{action}'.", "invalid_action", session_id)
        writer = expert_writer(session.expert_id) if session.expert_id else OTTO
        try:
            return await run(user_id, kwargs, writer, session_id)
        except (CollectionError, NotFoundError) as e:
            return _error(str(e), "refused", session_id)
        except ValidationError as e:
            return _error(_describe(e), "invalid_arguments", session_id)


async def _list(user_id: str, _a: dict, _w: str, session_id: str) -> ToolResponseBase:
    found = await collections_db().list_collections(user_id)
    return CollectionsResponse(
        message=f"{len(found)} collections.", session_id=session_id, collections=found
    ).from_outside()


async def _describe_one(
    user_id: str, args: dict, _w: str, session_id: str
) -> ToolResponseBase:
    collection = await _collection(user_id, args)
    page = await collections_db().query_rows(user_id, collection.id, RowQuery(limit=5))
    return _rows_response(
        f"'{collection.name}': {collection.row_count} rows; the newest five below.",
        session_id,
        collection,
        page.rows,
        page.total,
    )


async def _create(
    user_id: str, args: dict, _w: str, session_id: str
) -> ToolResponseBase:
    specs = [FieldSpec.model_validate(f) for f in _list_arg(args, "fields")]
    created = await collections_db().create_collection(
        user_id, _required(args, "collection"), specs, args.get("key_field") or None
    )
    return CollectionsResponse(
        message=f"Made '{created.name}'.", session_id=session_id, collection=created
    ).from_outside()


async def _add(
    user_id: str, args: dict, writer: str, session_id: str
) -> ToolResponseBase:
    rows = _list_arg(args, "rows")
    if not rows or len(rows) > MAX_ROWS_PER_CALL:
        raise CollectionError(f"Add 1 to {MAX_ROWS_PER_CALL} rows at a time")
    result = await collections_db().write_rows(
        user_id,
        _required(args, "collection"),
        rows,
        writer=writer,
        match_on_key=args.get("match_on_key") is True,
    )
    kept = (
        f" A person's edit was kept over your value in: {', '.join(sorted(result.held))}."
        if result.held
        else ""
    )
    return _rows_response(
        f"Added {result.added}, updated {result.updated} in "
        f"'{result.collection.name}'.{kept}",
        session_id,
        result.collection,
        result.rows,
        result.collection.row_count,
    )


async def _update(
    user_id: str, args: dict, writer: str, session_id: str
) -> ToolResponseBase:
    row = await collections_db().update_row(
        user_id,
        _required(args, "collection"),
        args.get("values") or {},
        writer=writer,
        key=args.get("key") or None,
        row_id=args.get("row_id") or None,
    )
    held = (
        f" Kept the person's value in: {', '.join(row.pending)}." if row.pending else ""
    )
    return _rows_response(f"Updated the row.{held}", session_id, None, [row], None)


async def _find(user_id: str, args: dict, _w: str, session_id: str) -> ToolResponseBase:
    find = _Find.model_validate(args.get("find") or {})
    collection = await _collection(user_id, args)
    page = await collections_db().query_rows(
        user_id,
        collection.id,
        RowQuery(conditions=find.conditions, sort=find.sort, limit=find.limit),
    )
    return _rows_response(
        f"{page.total} rows match; {len(page.rows)} shown.",
        session_id,
        collection,
        page.rows,
        page.total,
    )


async def _delete_rows(
    user_id: str, args: dict, _w: str, session_id: str
) -> ToolResponseBase:
    ids = [str(i) for i in _list_arg(args, "row_ids")][:MAX_ROWS_PER_CALL]
    deleted = await collections_db().delete_rows(
        user_id, _required(args, "collection"), row_ids=ids
    )
    return CollectionsResponse(
        message=f"Deleted {deleted} rows.", session_id=session_id
    ).from_outside()


async def _change_schema(
    user_id: str, args: dict, _w: str, session_id: str
) -> ToolResponseBase:
    change = SchemaChange.model_validate(args.get("change") or {})
    changed = await collections_db().change_schema(
        user_id, _required(args, "collection"), change
    )
    return CollectionsResponse(
        message=f"Changed '{changed.name}'.", session_id=session_id, collection=changed
    ).from_outside()


async def _delete_collection(
    user_id: str, args: dict, _w: str, session_id: str
) -> ToolResponseBase:
    name = _required(args, "collection")
    if not await collections_db().delete_collection(user_id, name):
        raise NotFoundError(f"No collection named '{name}'")
    return CollectionsResponse(
        message=f"Deleted '{name}' and its rows.", session_id=session_id
    ).from_outside()


_ACTIONS = {
    "list": _list,
    "describe": _describe_one,
    "create": _create,
    "add": _add,
    "update": _update,
    "find": _find,
    "delete_rows": _delete_rows,
    "change_schema": _change_schema,
    "delete_collection": _delete_collection,
}


def catalogue() -> dict[str, Any]:
    return {
        "actions": {
            "list": "Your collections, their fields and row counts.",
            "describe": "One collection's fields and its newest rows.",
            "create": "collection, fields [{name, type, options?, unit?}], key_field?",
            "add": "collection, rows, match_on_key?; makes the collection and any "
            "new field on first use",
            "update": "collection, key or row_id, values",
            "find": "collection, find {conditions [{field, op, value}], sort "
            "{field, direction}, limit}",
            "delete_rows": "collection, row_ids; asks the user first",
            "change_schema": "collection, change {rename?, add_fields?, "
            "remove_fields?, key_field?, clear_key?}; asks the user first",
            "delete_collection": "collection; asks the user first",
        },
        "field_types": [t.value for t in FieldType],
        "ops": [
            "is",
            "is_not",
            "contains",
            "gt",
            "lt",
            "gte",
            "lte",
            "empty",
            "not_empty",
        ],
        "limits": {
            "collections": MAX_COLLECTIONS_PER_USER,
            "rows": MAX_ROWS,
            "fields": MAX_FIELDS,
            "row_bytes": MAX_ROW_BYTES,
            "rows_per_call": MAX_ROWS_PER_CALL,
        },
        "notes": [
            "Dates are 2026-11-22 or 2026-11-22T14:00:00Z.",
            "A person's edit to a cell stands over a different value you write; "
            "the page offers them yours.",
        ],
    }


async def collections_disabled_tools(session: ChatSession) -> frozenset[str]:
    hidden = frozenset({COLLECTIONS_TOOL})
    if not session.user_id:
        return hidden
    enabled = await is_feature_enabled(Flag.COLLECTIONS, session.user_id)
    return frozenset() if enabled else hidden


async def _collection(user_id: str, args: dict) -> CollectionInfo:
    name = _required(args, "collection")
    found = await collections_db().get_collection(user_id, name)
    if found is None:
        raise NotFoundError(f"No collection named '{name}'")
    return found


def _rows_response(
    message: str,
    session_id: str,
    collection: CollectionInfo | None,
    rows: list[CollectionRowRecord],
    total: int | None,
) -> ToolResponseBase:
    return CollectionsResponse(
        message=message,
        session_id=session_id,
        collection=collection,
        rows=rows,
        total=total,
    ).from_outside([row.values for row in rows], [row.pending for row in rows])


def _required(args: dict, name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise CollectionError(f"Pass {name}")
    return value


def _list_arg(args: dict, name: str) -> list[Any]:
    value = args.get(name) or []
    if not isinstance(value, list):
        raise CollectionError(f"{name} is a list")
    return value


def _error(message: str, error: str, session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=message, error=error, session_id=session_id
    ).from_outside()


def _describe(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    ]
    return f"Invalid arguments. {'; '.join(problems[:5])}"
