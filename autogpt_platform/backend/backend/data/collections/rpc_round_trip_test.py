"""What the collection store returns over the DatabaseManager survives the trip
as models, not as plain dicts."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from backend.util.service import _build_return_adapter

from . import db
from .model import (
    CollectionField,
    CollectionInfo,
    CollectionRowRecord,
    FieldType,
    GroupValue,
    RowPage,
    WriteResult,
)

WHEN = datetime(2026, 11, 22, 9, tzinfo=UTC)
INFO = CollectionInfo(
    id="col-1",
    name="Invoices",
    fields=[
        CollectionField(name="invoice_no", type=FieldType.TEXT),
        CollectionField(name="status", type=FieldType.SELECT, options=["Booked"]),
    ],
    key_field="invoice_no",
    revision=3,
    row_count=12,
    updated_at=WHEN,
)
ROW = CollectionRowRecord(
    id="row-1",
    key="INV-7",
    values={"invoice_no": "INV-7", "amount": 340, "status": "Booked"},
    pending={"amount": 3500},
    manual_fields=["amount"],
    written_by="workflow:graph-1",
    revision=4,
    created_at=WHEN,
    updated_at=WHEN,
)


@pytest.mark.parametrize(
    "func, returns, value",
    [
        (db.get_collection, CollectionInfo | None, INFO),
        (db.create_collection, CollectionInfo, INFO),
        (db.change_schema, CollectionInfo, INFO),
        (db.list_collections, list[CollectionInfo], [INFO]),
        (db.update_row, CollectionRowRecord, ROW),
        (db.get_row, CollectionRowRecord | None, ROW),
        (db.edit_row, CollectionRowRecord, ROW),
        (db.add_row, CollectionRowRecord, ROW),
        (db.query_rows, RowPage, RowPage(rows=[ROW], total=1)),
        (
            db.group_rows,
            list[GroupValue],
            [GroupValue(label="Booked", value=9.0), GroupValue(label=None, value=None)],
        ),
        (
            db.write_rows,
            WriteResult,
            WriteResult(collection=INFO, rows=[ROW], updated=1, held={"amount": 1}),
        ),
    ],
)
def test_collection_models_survive_the_rpc_round_trip(
    func: Any, returns: Any, value: Any
) -> None:
    adapter = _build_return_adapter(func, returns, func.__name__)
    payload = json.loads(json.dumps(adapter.dump_python(value, mode="json")))

    assert adapter.validate_python(payload) == value
