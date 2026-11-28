"""The collection store against a real Postgres: writes, the who-wins rule,
concurrency, limits, ownership and query safety."""

import asyncio
import uuid
from typing import Any

import pytest

from backend.api.model import CreateGraph
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_graph
from backend.util.test import SpinTestServer

from . import db
from .model import (
    Aggregate,
    CollectionError,
    CollectionNotFoundError,
    Condition,
    FieldSpec,
    FieldType,
    RowQuery,
    Sort,
    StaleRowError,
    WriteResult,
    workflow_writer,
)
from .schema import SchemaChange

pytestmark = pytest.mark.asyncio(loop_scope="session")

INTAKE = workflow_writer("graph-intake")


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"collections-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Priya",
        }
    )
    return user.id


@pytest.fixture
async def user_id(server: SpinTestServer) -> str:
    return await _user()


async def _invoices(user_id: str) -> str:
    created = await db.create_collection(
        user_id,
        "Invoices",
        [
            FieldSpec(name="invoice_no", type=FieldType.TEXT),
            FieldSpec(name="supplier", type=FieldType.TEXT),
            FieldSpec(name="amount", type=FieldType.NUMBER),
            FieldSpec(name="status", type=FieldType.SELECT, options=["Booked"]),
        ],
        key_field="invoice_no",
    )
    return created.id


async def _upsert(user_id: str, values: dict[str, Any], **kw: Any) -> WriteResult:
    return await db.write_rows(
        user_id, "Invoices", [values], writer=INTAKE, match_on_key=True, **kw
    )


async def test_an_hourly_rerun_leaves_a_hand_corrected_amount_alone(user_id: str):
    collection = await _invoices(user_id)
    first = await _upsert(user_id, {"invoice_no": "INV-7", "amount": "3400.00"})
    row = first.rows[0]
    await db.edit_row(
        user_id, collection, row.id, {"amount": 340}, based_on=row.revision
    )

    again = await _upsert(
        user_id,
        {"invoice_no": "INV-7", "supplier": "Lumen Forklift Lease", "amount": 3400},
    )

    stored = await db.get_row(user_id, "invoices", key="INV-7")
    assert stored is not None
    assert stored.values == {
        "invoice_no": "INV-7",
        "amount": 340,
        "supplier": "Lumen Forklift Lease",
    }
    assert stored.pending == {}
    assert again.updated == 1 and again.added == 0


async def test_a_changed_source_value_over_a_hand_edit_waits_for_a_click(user_id: str):
    collection = await _invoices(user_id)
    row = (await _upsert(user_id, {"invoice_no": "INV-8", "amount": 3400})).rows[0]
    edited = await db.edit_row(
        user_id, collection, row.id, {"amount": 340}, based_on=row.revision
    )

    held = await _upsert(user_id, {"invoice_no": "INV-8", "amount": 3500})
    accepted = await db.accept_pending_values(
        user_id, collection, row.id, ["amount"], based_on=held.rows[0].revision
    )

    assert held.held == {"amount": 1}
    assert held.rows[0].values["amount"] == 340
    assert held.rows[0].pending == {"amount": 3500}
    assert edited.revision < held.rows[0].revision < accepted.revision
    assert accepted.values["amount"] == 3500 and accepted.pending == {}


async def test_a_page_edit_on_an_old_version_is_refused_with_the_current_row(
    user_id: str,
):
    collection = await _invoices(user_id)
    row = (await _upsert(user_id, {"invoice_no": "INV-9", "amount": 10})).rows[0]
    await _upsert(user_id, {"invoice_no": "INV-9", "amount": 11})

    with pytest.raises(StaleRowError) as refused:
        await db.edit_row(
            user_id, collection, row.id, {"amount": 12}, based_on=row.revision
        )

    assert refused.value.current is not None
    assert refused.value.current.values["amount"] == 11


async def test_a_refused_value_writes_nothing_from_its_batch(user_id: str):
    await _invoices(user_id)

    with pytest.raises(CollectionError, match="amount"):
        await db.write_rows(
            user_id,
            "Invoices",
            [{"invoice_no": "A", "amount": 1}, {"invoice_no": "B", "amount": "lots"}],
            writer=INTAKE,
        )

    page = await db.query_rows(user_id, "Invoices", RowQuery())
    assert page.total == 0


async def test_a_first_write_creates_the_collection_with_inferred_fields(user_id: str):
    result = await db.write_rows(
        user_id,
        "Follow-ups",
        [{"what": "Send Brightline the quote", "due": "2026-11-27", "done": False}],
        writer="otto",
    )

    types = {f.name: f.type for f in result.collection.fields}
    assert types == {
        "what": FieldType.TEXT,
        "due": FieldType.DATE,
        "done": FieldType.CHECKBOX,
    }
    assert result.collection.row_count == 1


async def test_a_retried_keyless_add_writes_once(user_id: str):
    for _ in range(2):
        await db.write_rows(
            user_id,
            "Competitor changes",
            [{"competitor": "Slotwise", "change": "raised its team plan"}],
            writer=INTAKE,
            write_id="node-exec-1",
        )

    page = await db.query_rows(user_id, "Competitor changes", RowQuery())
    assert page.total == 1


async def test_two_upserts_of_one_key_at_once_make_one_row_with_both_fields(
    user_id: str, monkeypatch: pytest.MonkeyPatch
):
    await _invoices(user_id)
    barrier = asyncio.Barrier(2)
    real = db._existing

    async def both_look_first(*args: Any, **kwargs: Any):
        # With the collection lock the second writer never gets here first,
        # so the barrier gives up and the first goes on alone.
        try:
            await asyncio.wait_for(barrier.wait(), timeout=3)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        return await real(*args, **kwargs)

    monkeypatch.setattr(db, "_existing", both_look_first)
    await asyncio.wait_for(
        asyncio.gather(
            _upsert(user_id, {"invoice_no": "INV-1", "supplier": "Nordkai Customs"}),
            _upsert(user_id, {"invoice_no": "INV-1", "amount": 1180.5}),
        ),
        timeout=60,
    )

    page = await db.query_rows(user_id, "Invoices", RowQuery())
    assert page.total == 1
    assert page.rows[0].values == {
        "invoice_no": "INV-1",
        "supplier": "Nordkai Customs",
        "amount": 1180.5,
    }


async def test_parallel_adds_at_the_row_limit_do_not_pass_it(
    user_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(db, "MAX_ROWS", 2)
    await _invoices(user_id)
    await _upsert(user_id, {"invoice_no": "INV-1"})
    barrier = asyncio.Barrier(2)
    real = db._existing

    async def both_counted_first(*args: Any, **kwargs: Any):
        # Each writer has counted the rows by now; unlocked, both saw one.
        try:
            await asyncio.wait_for(barrier.wait(), timeout=3)
        except (TimeoutError, asyncio.BrokenBarrierError):
            pass
        return await real(*args, **kwargs)

    monkeypatch.setattr(db, "_existing", both_counted_first)
    outcomes = await asyncio.wait_for(
        asyncio.gather(
            _upsert(user_id, {"invoice_no": "INV-2"}),
            _upsert(user_id, {"invoice_no": "INV-3"}),
            return_exceptions=True,
        ),
        timeout=60,
    )

    refused = [o for o in outcomes if isinstance(o, CollectionError)]
    assert len(refused) == 1
    assert "most rows" in str(refused[0])
    assert (await db.query_rows(user_id, "Invoices", RowQuery())).total == 2


@pytest.mark.parametrize(
    "limit, collection, refusal",
    [
        ("MAX_COLLECTIONS_PER_USER", "Another", "most collections"),
        ("MAX_ROWS", "Invoices", "most rows"),
    ],
)
async def test_a_write_past_a_limit_names_it(
    user_id: str,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    collection: str,
    refusal: str,
):
    monkeypatch.setattr(db, limit, 1)
    await _invoices(user_id)
    await _upsert(user_id, {"invoice_no": "INV-1"})

    with pytest.raises(CollectionError, match=refusal):
        await db.write_rows(
            user_id, collection, [{"invoice_no": "INV-2"}], writer=INTAKE
        )


async def test_an_oversized_row_is_refused(user_id: str):
    with pytest.raises(CollectionError, match="at most 8 KB"):
        await db.write_rows(
            user_id,
            "Notes",
            [{f"part{i}": "x" * 1_900 for i in range(5)}],
            writer=INTAKE,
        )


async def test_another_users_collection_answers_as_missing(user_id: str):
    collection = await _invoices(user_id)
    await _upsert(user_id, {"invoice_no": "INV-1", "amount": 5})
    stranger = await _user()

    assert await db.get_collection(stranger, collection) is None
    assert await db.get_collection(stranger, "Invoices") is None
    for call in (
        db.query_rows(stranger, collection, RowQuery()),
        db.get_row(stranger, "Invoices", key="INV-1"),
        db.aggregate_rows(stranger, collection, [Aggregate(op="count")]),
        db.edit_row(stranger, collection, "x", {"amount": 1}, based_on=1),
        db.update_row(stranger, "Invoices", {"amount": 1}, writer=INTAKE, key="INV-1"),
    ):
        with pytest.raises(CollectionNotFoundError):
            await call
    assert await db.delete_collection(stranger, collection) is False
    # Writing to a name the stranger lacks makes them their own collection.
    theirs = await db.write_rows(stranger, "Invoices", [{"x": 1}], writer=INTAKE)
    assert theirs.collection.id != collection
    assert (await db.get_row(user_id, "Invoices", key="INV-1")) is not None


HOSTILE = [
    'amount\'; DROP TABLE "CollectionRow"; --',
    'a" OR 1=1 --',
    "$.data ? (@ > 1)",
    '{"x": 1}',
    "50%_off\\",
]


@pytest.mark.parametrize("text", HOSTILE)
async def test_field_names_and_values_are_data_not_sql(user_id: str, text: str):
    name = text[:60]
    await db.write_rows(
        user_id,
        "Odd",
        [{name: text, "n": 1}, {name: "plain", "n": 2}],
        writer=INTAKE,
    )

    found = await db.query_rows(
        user_id,
        "Odd",
        RowQuery(
            conditions=[Condition(field=name, op="is", value=text)],
            search=text,
            sort=Sort(field=name, direction="desc"),
        ),
    )
    total = await db.aggregate_rows(
        user_id,
        "Odd",
        [Aggregate(op="sum", field="n")],
        [Condition(field=name, op="contains", value=text)],
    )

    assert [row.values["n"] for row in found.rows] == [1]
    assert total == [1.0]
    assert (await db.query_rows(user_id, "Odd", RowQuery())).total == 2


async def test_filters_sort_search_chips_and_aggregates(user_id: str):
    await _invoices(user_id)
    for no, supplier, amount, status in [
        ("INV-1", "Harbour Pallets BV", 640.0, "Booked"),
        ("INV-2", "Nordkai Customs", 1180.5, "Booked"),
        ("INV-3", "Lumen Forklift Lease", 3400.0, "Needs review"),
        ("INV-4", "Harbour Pallets BV", 2210.0, "Needs review"),
    ]:
        await _upsert(
            user_id,
            {
                "invoice_no": no,
                "supplier": supplier,
                "amount": amount,
                "status": status,
            },
        )

    over = await db.query_rows(
        user_id,
        "Invoices",
        RowQuery(
            conditions=[Condition(field="amount", op="gt", value=1000)],
            sort=Sort(field="amount", direction="desc"),
        ),
    )
    harbour = await db.query_rows(user_id, "Invoices", RowQuery(search="harbour"))
    review = await db.query_rows(
        user_id, "Invoices", RowQuery(chips={"status": ["needs review"]})
    )
    numbers = await db.aggregate_rows(
        user_id,
        "Invoices",
        [
            Aggregate(op="count"),
            Aggregate(op="sum", field="amount"),
            Aggregate(op="max", field="amount"),
        ],
        [Condition(field="status", op="is", value="Booked")],
    )

    assert [r.key for r in over.rows] == ["INV-3", "INV-4", "INV-2"]
    assert harbour.total == 2
    assert {r.key for r in review.rows} == {"INV-3", "INV-4"}
    assert numbers == [2.0, 1820.5, 1180.5]


async def test_the_owner_changes_the_key_fields_and_name(user_id: str):
    collection = await _invoices(user_id)
    await _upsert(
        user_id, {"invoice_no": "INV-1", "supplier": "Nordkai Customs", "amount": 5}
    )
    await _upsert(
        user_id, {"invoice_no": "INV-2", "supplier": "Nordkai Customs", "amount": 6}
    )

    with pytest.raises(CollectionError, match="cannot be the key"):
        await db.change_schema(user_id, collection, SchemaChange(key_field="supplier"))
    changed = await db.change_schema(
        user_id,
        collection,
        SchemaChange(
            rename="Supplier invoices",
            remove_fields=["supplier"],
            add_fields=[FieldSpec(name="received", type=FieldType.DATE)],
            key_field="amount",
        ),
    )

    assert changed.name == "Supplier invoices"
    assert [f.name for f in changed.fields] == [
        "invoice_no",
        "amount",
        "status",
        "received",
    ]
    row = await db.get_row(user_id, "supplier invoices", key="6")
    assert row is not None and "supplier" not in row.values


async def test_only_the_users_own_workflows_count_as_made_by_them(
    user_id: str, server: SpinTestServer
):
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user_id
    )

    assert await db.graph_created_by(user_id, graph.id) is True
    assert await db.graph_created_by(await _user(), graph.id) is False
