"""The owner's view of their collections: read, edit, add, delete and reshape."""

import csv
import io
from typing import Annotated, Any, Literal

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Depends, Path, Query, Response, Security
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.collections import db
from backend.data.collections.model import (
    MAX_COLLECTIONS_PER_USER,
    MAX_FIELD_NAME_CHARS,
    MAX_FIELDS,
    MAX_ROW_BYTES,
    MAX_ROWS,
    CollectionInfo,
    CollectionRowRecord,
    Condition,
    RowPage,
    RowQuery,
    Sort,
    StaleRowError,
)
from backend.data.collections.schema import SchemaChange
from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

router = APIRouter(
    prefix="/collections",
    tags=["collections"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)

CollectionID = Annotated[str, Path(max_length=64)]
RowID = Annotated[str, Path(max_length=64)]
FieldName = Annotated[str, Field(min_length=1, max_length=MAX_FIELD_NAME_CHARS)]


class CollectionLimits(BaseModel):
    collections: int
    rows: int
    fields: int
    row_bytes: int


def _limits() -> CollectionLimits:
    return CollectionLimits(
        collections=MAX_COLLECTIONS_PER_USER,
        rows=MAX_ROWS,
        fields=MAX_FIELDS,
        row_bytes=MAX_ROW_BYTES,
    )


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo]
    limits: CollectionLimits = Field(default_factory=_limits)


class CollectionResponse(BaseModel):
    collection: CollectionInfo
    limits: CollectionLimits = Field(default_factory=_limits)


class RowFilterRequest(BaseModel):
    conditions: list[Condition] = Field(default_factory=list, max_length=3)
    chips: dict[FieldName, list[Annotated[str, Field(max_length=100)]]] = Field(
        default_factory=dict, max_length=3
    )


class RowAddRequest(BaseModel):
    values: dict[FieldName, Any] = Field(max_length=MAX_FIELDS)


class RowEditRequest(BaseModel):
    values: dict[FieldName, Any] = Field(min_length=1, max_length=MAX_FIELDS)
    based_on: int = Field(description="The row's revision the edit was made on.")


class AcceptPendingRequest(BaseModel):
    fields: list[FieldName] = Field(min_length=1, max_length=MAX_FIELDS)
    based_on: int


class CollectionUpdateRequest(SchemaChange):
    pass


class StaleRow(BaseModel):
    detail: str
    row: CollectionRowRecord | None = None


@router.get("", operation_id="list_collections")
async def list_collections(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> CollectionListResponse:
    await _require_flag(user_id)
    return CollectionListResponse(collections=await db.list_collections(user_id))


@router.get("/{collection_id}", operation_id="get_collection")
async def get_collection(
    collection_id: CollectionID,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> CollectionResponse:
    await _require_flag(user_id)
    return CollectionResponse(collection=await _owned(user_id, collection_id))


@router.get("/{collection_id}/rows", operation_id="list_collection_rows")
async def list_rows(
    collection_id: CollectionID,
    search: Annotated[str | None, Query(max_length=200)] = None,
    sort: Annotated[str | None, Query(max_length=MAX_FIELD_NAME_CHARS)] = None,
    direction: Literal["asc", "desc"] = "asc",
    where: Annotated[
        str | None,
        Query(max_length=4_000, description="A RowFilterRequest as JSON."),
    ] = None,
    offset: Annotated[int, Query(ge=0, le=MAX_ROWS)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> RowPage:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    try:
        parsed = RowFilterRequest.model_validate_json(where) if where else None
    except ValidationError:
        raise ValueError("The filter is not a valid row filter") from None
    query = RowQuery(
        conditions=parsed.conditions if parsed else [],
        chips=parsed.chips if parsed else {},
        search=search,
        sort=Sort(field=sort, direction=direction) if sort else None,
        offset=offset,
        limit=limit,
    )
    return await db.query_rows(user_id, owned.id, query)


@router.get(
    "/{collection_id}/export",
    operation_id="export_collection",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
)
async def export_collection(
    collection_id: CollectionID,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    """Every row as CSV, one column per field, in the collection's order."""
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    rows = []
    while len(rows) < owned.row_count:
        page = await db.query_rows(
            user_id, owned.id, RowQuery(limit=200, offset=len(rows))
        )
        if not page.rows:
            break
        rows += page.rows
    return Response(
        content=rows_csv(owned, rows),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{_file_name(owned.name)}.csv"'
        },
    )


@router.post(
    "/{collection_id}/rows",
    operation_id="add_collection_row",
    dependencies=[Depends(enforce_payment_paywall)],
)
async def add_row(
    collection_id: CollectionID,
    body: RowAddRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> CollectionRowRecord:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    return await db.add_row(user_id, owned.id, body.values)


@router.patch(
    "/{collection_id}/rows/{row_id}",
    operation_id="edit_collection_row",
    response_model=CollectionRowRecord,
    responses={409: {"model": StaleRow}},
)
async def edit_row(
    collection_id: CollectionID,
    row_id: RowID,
    body: RowEditRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Any:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    try:
        return await db.edit_row(
            user_id, owned.id, row_id, body.values, based_on=body.based_on
        )
    except StaleRowError as e:
        return _stale(e)


@router.post(
    "/{collection_id}/rows/{row_id}/accept",
    operation_id="accept_collection_row_values",
    response_model=CollectionRowRecord,
    responses={409: {"model": StaleRow}},
)
async def accept_pending(
    collection_id: CollectionID,
    row_id: RowID,
    body: AcceptPendingRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Any:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    try:
        return await db.accept_pending_values(
            user_id, owned.id, row_id, body.fields, based_on=body.based_on
        )
    except StaleRowError as e:
        return _stale(e)


@router.delete(
    "/{collection_id}/rows/{row_id}",
    operation_id="delete_collection_row",
    status_code=204,
)
async def delete_row(
    collection_id: CollectionID,
    row_id: RowID,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    if not await db.delete_rows(user_id, owned.id, row_ids=[row_id]):
        raise NotFoundError(f"Row #{row_id} not found")
    return Response(status_code=204)


@router.patch("/{collection_id}", operation_id="update_collection")
async def update_collection(
    collection_id: CollectionID,
    body: CollectionUpdateRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> CollectionResponse:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    change = SchemaChange.model_validate(body.model_dump())
    return CollectionResponse(
        collection=await db.change_schema(user_id, owned.id, change)
    )


@router.delete("/{collection_id}", operation_id="delete_collection", status_code=204)
async def delete_collection(
    collection_id: CollectionID,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    await _require_flag(user_id)
    owned = await _owned(user_id, collection_id)
    await db.delete_collection(user_id, owned.id)
    return Response(status_code=204)


def _stale(error: StaleRowError) -> JSONResponse:
    """409 with the row as it is now, so the page shows the new values."""
    body = StaleRow(detail=str(error), row=error.current)
    return JSONResponse(status_code=409, content=body.model_dump(mode="json"))


def rows_csv(collection: CollectionInfo, rows: list[CollectionRowRecord]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    names = [field.name for field in collection.fields]
    writer.writerow(names)
    for row in rows:
        writer.writerow([_csv_cell(row.values.get(name)) for name in names])
    return out.getvalue()


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        text = ", ".join(str(v) for v in value)
    else:
        text = str(value)
    # A spreadsheet runs a cell that starts like a formula; this one is data.
    return f"'{text}" if text[:1] in _FORMULA_STARTS else text


_FORMULA_STARTS = {"=", "+", "-", "@", "\t", "\r"}


def _file_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return safe or "collection"


async def _owned(user_id: str, collection_id: str) -> CollectionInfo:
    """The caller's collection by id; a name or another user's id answers 404."""
    found = await db.get_collection(user_id, collection_id)
    if found is None or found.id != collection_id:
        raise NotFoundError(f"Collection #{collection_id} not found")
    return found


async def _require_flag(user_id: str) -> None:
    if not await is_feature_enabled(Flag.COLLECTIONS, user_id, default=False):
        raise NotFoundError("Not found")
