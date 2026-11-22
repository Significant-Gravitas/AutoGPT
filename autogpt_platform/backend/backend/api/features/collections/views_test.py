"""Board, cards and chart sections, row buttons, and the CSV export, against Postgres."""

import csv
import io
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import fastapi
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from httpx import ASGITransport, AsyncClient

from backend.api.features.library import db as library_db
from backend.api.features.pages.design import PageDesign
from backend.api.features.pages.resolved import (
    PageCollectionBoard,
    PageCollectionCards,
    PageCollectionChart,
)
from backend.api.features.pages.service import check, resolve_page
from backend.api.model import CreateGraph
from backend.api.rest_api import app as rest_app
from backend.data.collections import db as collections_db
from backend.data.collections.model import FieldSpec, FieldType, workflow_writer
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_graph
from backend.util.test import SpinTestServer

from . import routes

pytestmark = pytest.mark.asyncio(loop_scope="session")
SOURCES_FLAG = "backend.api.features.pages.sources.is_feature_enabled"


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"board-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Jonas",
        }
    )
    return user.id


@pytest.fixture
async def desk(server: SpinTestServer) -> dict[str, str]:
    """Jonas's content ideas, and a workflow of his to put on the page."""
    user_id = await _user()
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user_id
    )
    agent = await library_db.get_library_agent_by_graph_id(user_id, graph.id)
    assert agent is not None
    created = await collections_db.create_collection(
        user_id,
        "Content ideas",
        [
            FieldSpec(name="keyword", type=FieldType.TEXT),
            FieldSpec(name="volume", type=FieldType.NUMBER),
            FieldSpec(
                name="status",
                type=FieldType.SELECT,
                options=["New", "Drafted", "Posted"],
            ),
            FieldSpec(name="posted_on", type=FieldType.DATE),
        ],
        key_field="keyword",
    )
    await collections_db.write_rows(
        user_id,
        created.id,
        [
            {"keyword": "cohort retention chart", "volume": 1900, "status": "New"},
            {"keyword": "cohort analysis example", "volume": 1300, "status": "Drafted"},
            {
                "keyword": "weekly cohort table",
                "volume": 480,
                "status": "Posted",
                "posted_on": "2026-11-03",
            },
            {
                "keyword": "cohort churn formula",
                "volume": 390,
                "status": "Posted",
                "posted_on": "2026-11-12",
            },
        ],
        writer=workflow_writer(graph.id),
    )
    return {
        "user_id": user_id,
        "collection": created.id,
        "workflow": agent.id,
        "graph": graph.id,
    }


def _design(desk: dict[str, str], *sections: dict[str, Any]) -> PageDesign:
    return PageDesign.model_validate(
        {
            "title": "Content desk",
            "sections": [
                *sections,
                {
                    "kind": "run_form",
                    "workflow": desk["workflow"],
                    "fields": [{"input": "input_1"}, {"input": "input_2"}],
                },
            ],
        }
    )


async def _resolve(desk: dict[str, str], design: PageDesign):
    with patch(SOURCES_FLAG, AsyncMock(return_value=True)):
        sources, problems = await check(desk["user_id"], design)
        return await resolve_page(desk["user_id"], design, sources), problems


async def test_a_board_shows_a_column_per_option_with_its_draft_button(desk):
    source = {"type": "collection", "id": desk["collection"]}
    button = {
        "label": "Draft",
        "workflow": desk["workflow"],
        "inputs": {"input_1": "keyword"},
    }

    page, problems = await _resolve(
        desk,
        _design(
            desk,
            {
                "kind": "collection_board",
                "source": source,
                "group_by": "status",
                "card_title": "keyword",
                "card_fields": ["volume"],
                "buttons": [button],
            },
        ),
    )

    board = page.sections[0]
    assert problems == []
    assert isinstance(board, PageCollectionBoard)
    assert [(c.option, c.total) for c in board.columns] == [
        ("New", 1),
        ("Drafted", 1),
        ("Posted", 2),
    ]
    assert board.columns[0].rows[0].values == {
        "keyword": "cohort retention chart",
        "volume": 1900,
        "status": "New",
    }
    assert [(b.label, b.graph_id, b.inputs) for b in board.buttons] == [
        ("Draft", desk["graph"], {"input_1": "keyword"})
    ]


async def test_cards_and_a_chart_read_the_same_rows(desk):
    source = {"type": "collection", "id": desk["collection"]}

    page, problems = await _resolve(
        desk,
        _design(
            desk,
            {
                "kind": "collection_cards",
                "source": source,
                "card_title": "keyword",
                "card_fields": ["volume", "status"],
                "sort": {"field": "volume", "direction": "desc"},
                "limit": 3,
            },
            {
                "kind": "collection_chart",
                "source": source,
                "value": "sum",
                "value_field": "volume",
                "group_by": "status",
            },
            {
                "kind": "collection_chart",
                "source": source,
                "group_by": "posted_on",
                "bucket": "week",
                "chart": "line",
                "filter": [{"field": "posted_on", "op": "not_empty"}],
            },
        ),
    )

    cards, by_status, by_week = page.sections[:3]
    assert problems == []
    assert isinstance(cards, PageCollectionCards)
    assert [r.values["keyword"] for r in cards.rows] == [
        "cohort retention chart",
        "cohort analysis example",
        "weekly cohort table",
    ]
    assert cards.total == 4
    assert isinstance(by_status, PageCollectionChart) and by_status.block
    assert by_status.block.x_labels == ["New", "Drafted", "Posted"]
    assert by_status.block.series[0].values == [1900.0, 1300.0, 870.0]
    assert isinstance(by_week, PageCollectionChart) and by_week.block
    assert by_week.block.x_labels == ["Week of 2026-11-02", "Week of 2026-11-09"]


@pytest.mark.parametrize(
    "section, message",
    [
        (
            {
                "kind": "collection_board",
                "group_by": "keyword",
                "card_title": "keyword",
            },
            "a board groups by a select field",
        ),
        (
            {"kind": "collection_chart", "group_by": "volume"},
            "a chart groups by a select or a date field",
        ),
        (
            {
                "kind": "collection_cards",
                "card_title": "keyword",
                "buttons": [
                    {
                        "label": "Go",
                        "workflow": "WORKFLOW",
                        "inputs": {"topic": "keyword"},
                    }
                ],
            },
            "has no input named 'topic'",
        ),
        (
            {
                "kind": "collection_cards",
                "card_title": "keyword",
                "buttons": [
                    {
                        "label": "Go",
                        "workflow": "elsewhere",
                        "inputs": {"input_1": "keyword"},
                    }
                ],
            },
            "names a workflow that has no section on this page",
        ),
    ],
    ids=["board on text", "chart on number", "missing input", "workflow off the page"],
)
async def test_a_section_that_does_not_fit_says_why(desk, section, message):
    for button in section.get("buttons", []):
        if button["workflow"] == "WORKFLOW":
            button["workflow"] = desk["workflow"]
    source = {"type": "collection", "id": desk["collection"]}

    _, problems = await _resolve(desk, _design(desk, {**section, "source": source}))

    assert any(message in p.message for p in problems), problems


async def test_the_export_is_every_row_with_formulas_made_inert(desk):
    user_id = desk["user_id"]
    await collections_db.write_rows(
        user_id,
        desk["collection"],
        [{"keyword": '=HYPERLINK("x")', "volume": -40, "status": "New"}],
        writer="otto",
    )
    app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
    app.include_router(routes.router, prefix="/api")
    as_user = {"id": user_id}
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": as_user["id"],
        "role": "user",
        "email": "jonas@example.com",
    }

    with patch.object(routes, "is_feature_enabled", AsyncMock(return_value=True)):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http:
            mine = await http.get(f"/api/collections/{desk['collection']}/export")
            as_user["id"] = await _user()
            theirs = await http.get(f"/api/collections/{desk['collection']}/export")

    rows = list(csv.reader(io.StringIO(mine.text)))
    assert mine.status_code == 200
    assert mine.headers["content-type"].startswith("text/csv")
    assert 'filename="Content ideas.csv"' in mine.headers["content-disposition"]
    assert rows[0] == ["keyword", "volume", "status", "posted_on"]
    assert len(rows) == 6
    assert ['\'=HYPERLINK("x")', "-40", "New", ""] in rows
    assert theirs.status_code == 404
