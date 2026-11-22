"""Collections end to end against Postgres: the routes through the app's own
error handlers, Otto's tool through ``BaseTool.execute``, the blocks' guard,
and the page sections that show a collection."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import fastapi
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from httpx import ASGITransport, AsyncClient

from backend.api.features.pages.design import PageDesign
from backend.api.features.pages.resolved import (
    PageCollectionNumbers,
    PageCollectionTable,
    PageNotice,
)
from backend.api.features.pages.service import check, resolve_page
from backend.api.model import CreateGraph
from backend.api.rest_api import app as rest_app
from backend.blocks.collections import (
    INSTALLED_REFUSAL,
    AddToCollectionBlock,
    FindCollectionRowsBlock,
    GetCollectionRowBlock,
    UpdateCollectionRowBlock,
)
from backend.copilot.gate.policy import Effect, effect_for, is_irreversible
from backend.copilot.model import ChatSession, create_chat_session
from backend.copilot.tools.collections_tool import CollectionsTool
from backend.data.collections import db as collections_db
from backend.data.collections.model import (
    CollectionError,
    FieldSpec,
    FieldType,
    RowQuery,
    workflow_writer,
)
from backend.data.execution import ExecutionContext
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_graph
from backend.util.test import SpinTestServer

from . import routes

pytestmark = pytest.mark.asyncio(loop_scope="session")
FLAG = "backend.util.feature_flag.is_feature_enabled"


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"coll-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Sam",
        }
    )
    return user.id


@pytest.fixture
async def ledger(server: SpinTestServer) -> tuple[str, str]:
    """A user and their invoice ledger with three invented invoices."""
    user_id = await _user()
    created = await collections_db.create_collection(
        user_id,
        "Invoices",
        [
            FieldSpec(name="invoice_no", type=FieldType.TEXT),
            FieldSpec(name="supplier", type=FieldType.TEXT),
            FieldSpec(name="amount", type=FieldType.NUMBER, unit="$"),
            FieldSpec(name="status", type=FieldType.SELECT, options=["Booked"]),
        ],
        key_field="invoice_no",
    )
    await collections_db.write_rows(
        user_id,
        created.id,
        [
            {
                "invoice_no": "INV-1",
                "supplier": "Harbour Pallets BV",
                "amount": 640,
                "status": "Booked",
            },
            {
                "invoice_no": "INV-2",
                "supplier": "Nordkai Customs",
                "amount": 1180.5,
                "status": "Booked",
            },
            {
                "invoice_no": "INV-3",
                "supplier": "Lumen Forklift Lease",
                "amount": 3400,
                "status": "Needs review",
            },
        ],
        writer=workflow_writer("graph-intake"),
    )
    return user_id, created.id


@pytest.fixture
async def client(
    ledger: tuple[str, str],
) -> AsyncIterator[tuple[AsyncClient, dict[str, Any]]]:
    app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
    app.include_router(routes.router, prefix="/api")
    as_user = {"id": ledger[0], "flag": True}
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": as_user["id"],
        "role": "user",
        "email": "sam@example.com",
    }

    async def flag(*_args: Any, **_kwargs: Any) -> bool:
        return bool(as_user["flag"])

    with patch.object(routes, "is_feature_enabled", flag):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http:
            yield http, as_user


async def test_the_owner_reads_adds_edits_accepts_and_deletes(client, ledger):
    http, _ = client
    user_id, collection = ledger
    base = f"/api/collections/{collection}"

    listed = (await http.get("/api/collections")).json()
    found = (await http.get(f"{base}/rows", params={"search": "harbour"})).json()
    added = await http.post(
        f"{base}/rows", json={"values": {"invoice_no": "INV-9", "amount": 70}}
    )
    row = found["rows"][0]
    edited = await http.patch(
        f"{base}/rows/{row['id']}",
        json={"values": {"amount": 64}, "based_on": row["revision"]},
    )
    stale = await http.patch(
        f"{base}/rows/{row['id']}",
        json={"values": {"amount": 65}, "based_on": row["revision"]},
    )
    held = await collections_db.write_rows(
        user_id,
        collection,
        [{"invoice_no": "INV-1", "amount": 650}],
        writer=workflow_writer("graph-intake"),
        match_on_key=True,
    )
    accepted = await http.post(
        f"{base}/rows/{row['id']}/accept",
        json={"fields": ["amount"], "based_on": held.rows[0].revision},
    )
    deleted = await http.delete(f"{base}/rows/{row['id']}")
    reshaped = await http.patch(base, json={"rename": "Supplier invoices"})
    gone = await http.delete(base)

    assert [c["name"] for c in listed["collections"]] == ["Invoices"]
    assert listed["limits"]["rows"] == 5_000
    assert found["total"] == 1
    assert added.status_code == 200 and added.json()["manual_fields"] == [
        "amount",
        "invoice_no",
    ]
    assert edited.status_code == 200 and edited.json()["values"]["amount"] == 64
    assert stale.status_code == 409 and stale.json()["row"]["values"]["amount"] == 64
    assert held.rows[0].pending == {"amount": 650}
    assert accepted.status_code == 200 and accepted.json()["values"]["amount"] == 650
    assert deleted.status_code == 204
    assert reshaped.json()["collection"]["name"] == "Supplier invoices"
    assert gone.status_code == 204
    assert await collections_db.get_collection(user_id, collection) is None


async def test_another_user_and_a_name_in_place_of_an_id_answer_404(client, ledger):
    http, as_user = client
    _, collection = ledger
    base = f"/api/collections/{collection}"
    calls = [
        ("GET", base, None),
        ("GET", f"{base}/rows", None),
        ("POST", f"{base}/rows", {"values": {"invoice_no": "X"}}),
        ("PATCH", f"{base}/rows/x", {"values": {"amount": 1}, "based_on": 1}),
        ("POST", f"{base}/rows/x/accept", {"fields": ["amount"], "based_on": 1}),
        ("DELETE", f"{base}/rows/x", None),
        ("PATCH", base, {"rename": "Mine now"}),
        ("DELETE", base, None),
    ]

    as_user["id"] = await _user()
    stranger = [(await http.request(m, u, json=b)).status_code for m, u, b in calls]
    as_user["id"] = ledger[0]
    by_name = (await http.get("/api/collections/Invoices")).status_code
    as_user["flag"] = False
    flag_off = (await http.get("/api/collections")).status_code

    assert stranger == [404] * len(calls)
    assert by_name == 404
    assert flag_off == 404
    assert (
        await collections_db.query_rows(ledger[0], collection, RowQuery())
    ).total == 3


async def test_the_tool_does_each_action_for_otto_and_experts(ledger):
    user_id, collection = ledger
    tool = CollectionsTool()
    session = await create_chat_session(user_id, dry_run=False)

    async def call(**kwargs: Any) -> dict[str, Any]:
        result = await tool.execute(
            user_id, session, f"c-{uuid.uuid4().hex[:6]}", **kwargs
        )
        output = result.output
        return json.loads(output) if isinstance(output, str) else output

    with patch(
        f"{CollectionsTool.__module__}.is_feature_enabled", AsyncMock(return_value=True)
    ):
        catalogue = await call()
        listed = await call(action="list")
        added = await call(
            action="add",
            collection="Follow-ups",
            rows=[
                {
                    "what": "Send Brightline the revised quote",
                    "due": "2026-11-27",
                    "done": False,
                },
                {
                    "what": "Book Oko Studio's kickoff",
                    "due": "2026-12-03",
                    "done": False,
                },
            ],
        )
        found = await call(
            action="find",
            collection="Follow-ups",
            find={"conditions": [{"field": "due", "op": "lte", "value": "2026-11-30"}]},
        )
        updated = await call(
            action="update",
            collection="Invoices",
            key="INV-3",
            values={"status": "Booked"},
        )
        described = await call(action="describe", collection=collection)
        refused = await call(
            action="add", collection="Invoices", rows=[{"amount": "lots"}]
        )
        reshaped = await call(
            action="change_schema",
            collection="Follow-ups",
            change={"key_field": "what"},
        )
        removed = await call(
            action="delete_rows",
            collection="Invoices",
            row_ids=[found["rows"][0]["id"]],
        )
        dropped = await call(action="delete_collection", collection="Follow-ups")

    assert "actions" in catalogue["catalogue"]
    assert [c["name"] for c in listed["collections"]] == ["Invoices"]
    assert added["collection"]["row_count"] == 2
    assert {f["name"]: f["type"] for f in added["collection"]["fields"]} == {
        "what": "text",
        "due": "date",
        "done": "checkbox",
    }
    assert found["total"] == 1 and found["rows"][0]["written_by"] == "otto"
    assert updated["rows"][0]["values"]["status"] == "Booked"
    assert described["total"] == 3
    assert refused["error"] == "refused" and "amount" in refused["message"]
    assert reshaped["collection"]["key_field"] == "what"
    assert removed["message"] == "Deleted 0 rows."
    assert dropped["message"].startswith("Deleted 'Follow-ups'")


async def test_reads_ask_nothing_row_writes_are_workspace_and_the_rest_asks(ledger):
    user_id, _ = ledger
    tool = CollectionsTool()
    session = ChatSession.new(user_id=user_id, dry_run=False)

    effects = {
        action: await tool.gate_subject(user_id, session, {"action": action})
        for action in [
            "find",
            "add",
            "update",
            "delete_rows",
            "change_schema",
            "delete_collection",
        ]
    }

    assert effects["find"] is not None and effects["find"].effect is Effect.READ
    assert effects["add"] is not None and effects["add"].effect is Effect.WORKSPACE
    assert (
        effects["update"] is not None and effects["update"].effect is Effect.WORKSPACE
    )
    assert [
        effects[a] for a in ("delete_rows", "change_schema", "delete_collection")
    ] == [None] * 3
    assert effect_for("collections") is Effect.PLATFORM
    assert is_irreversible("collections", {"action": "delete_collection"}) is True


async def test_a_block_in_an_installed_workflow_stops_and_writes_nothing(
    server: SpinTestServer, ledger
):
    user_id, collection = ledger
    author = await _user()
    theirs = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), author
    )
    mine = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user_id
    )
    block = AddToCollectionBlock()

    async def run(graph_id: str) -> list[tuple[str, Any]]:
        data = block.input_schema(
            collection="Invoices", row={"invoice_no": f"G-{graph_id[:4]}"}
        )
        return [
            output
            async for output in block.run(
                data,
                user_id=user_id,
                graph_id=graph_id,
                node_exec_id=str(uuid.uuid4()),
                execution_context=ExecutionContext(user_id=user_id),
            )
        ]

    with patch(
        "backend.blocks.collections.is_feature_enabled", AsyncMock(return_value=True)
    ):
        with pytest.raises(CollectionError, match="installed"):
            await run(theirs.id)
        written = dict(await run(mine.id))

    assert INSTALLED_REFUSAL.startswith("Collection blocks run only")
    assert written["added"] is True
    rows = await collections_db.query_rows(user_id, collection, RowQuery())
    assert rows.total == 4
    assert rows.rows[0].written_by == workflow_writer(mine.id)


async def test_page_sections_read_the_collection_for_its_owner(ledger):
    user_id, collection = ledger
    source = {"type": "collection", "id": collection}
    design = PageDesign.model_validate(
        {
            "title": "Invoices",
            "sections": [
                {
                    "kind": "collection_numbers",
                    "source": source,
                    "numbers": [
                        {
                            "label": "Booked",
                            "op": "sum",
                            "field": "amount",
                            "filter": [
                                {"field": "status", "op": "is", "value": "Booked"}
                            ],
                        },
                        {
                            "label": "Needs review",
                            "op": "count",
                            "filter": [
                                {"field": "status", "op": "is", "value": "Needs review"}
                            ],
                        },
                    ],
                },
                {
                    "kind": "collection_table",
                    "source": source,
                    "columns": ["invoice_no", "amount", "approved_by"],
                    "sort": {"field": "amount", "direction": "desc"},
                    "editable": False,
                },
                {
                    "kind": "collection_table",
                    "source": source,
                    "columns": ["invoice_no"],
                    "filter": [{"field": "received", "op": "empty"}],
                },
            ],
        }
    )

    with patch(
        "backend.api.features.pages.sources.is_feature_enabled",
        AsyncMock(return_value=True),
    ):
        sources, problems = await check(user_id, design)
        page = await resolve_page(user_id, design, sources)
        stranger_sources, stranger_problems = await check(await _user(), design)

    numbers, table, stale = page.sections
    assert isinstance(numbers, PageCollectionNumbers)
    assert [(i.label, i.value) for i in numbers.block.items] == [
        ("Booked", "$1,820.50"),
        ("Needs review", "1"),
    ]
    assert isinstance(table, PageCollectionTable)
    assert [c.field for c in table.columns] == ["invoice_no", "amount"]
    assert table.dropped_columns == ["approved_by"]
    assert [r.values["invoice_no"] for r in table.rows] == ["INV-3", "INV-2", "INV-1"]
    assert table.read_only is True and "supplier" not in table.rows[0].values
    assert isinstance(stale, PageNotice) and stale.reason == "stale"
    assert [p.section for p in problems] == [2]
    assert {p.message for p in stranger_problems} == {
        "its collection is no longer available"
    }


async def test_a_workflow_adds_finds_and_updates_through_the_blocks(
    server: SpinTestServer,
):
    """Keyword ideas added twice by key, the new ones found, one drafted:
    each block fed the one before's output, against Postgres."""
    user_id = await _user()
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user_id
    )
    context = ExecutionContext(user_id=user_id, graph_id=graph.id)

    async def run(block: Any, **inputs: Any) -> list[tuple[str, Any]]:
        return [
            output
            async for output in block.run(
                block.input_schema(**inputs),
                user_id=user_id,
                graph_id=graph.id,
                node_exec_id=str(uuid.uuid4()),
                execution_context=context,
            )
        ]

    ideas = [("cohort retention chart", 1900), ("weekly cohort table", 480)]
    with patch(
        "backend.blocks.collections.is_feature_enabled", AsyncMock(return_value=True)
    ):
        await collections_db.create_collection(
            user_id,
            "Content ideas",
            [FieldSpec(name="keyword", type=FieldType.TEXT)],
            key_field="keyword",
        )
        for _ in range(2):
            for keyword, volume in ideas:
                await run(
                    AddToCollectionBlock(),
                    collection="Content ideas",
                    row={"keyword": keyword, "volume": volume, "status": "New"},
                    match_on_key=True,
                )
        found = dict(
            await run(
                FindCollectionRowsBlock(),
                collection="Content ideas",
                conditions=[{"field": "volume", "op": "gt", "value": 1000}],
            )
        )
        pick = found["rows"][0]
        await run(
            UpdateCollectionRowBlock(),
            collection="Content ideas",
            row_id=pick["_id"],
            values={"status": "Drafted", "draft": "What nobody tells you"},
        )
        read = dict(
            await run(
                GetCollectionRowBlock(),
                collection="Content ideas",
                key="cohort retention chart",
            )
        )

    assert found["count"] == 1 and pick["keyword"] == "cohort retention chart"
    assert read["row"]["status"] == "Drafted"
    assert read["row"]["draft"] == "What nobody tells you"
    assert (
        await collections_db.query_rows(user_id, "Content ideas", RowQuery())
    ).total == 2
