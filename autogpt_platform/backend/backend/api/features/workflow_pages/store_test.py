"""Workflow pages against the database: storage, the routes, and Otto's tool."""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import fastapi
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from httpx import ASGITransport, AsyncClient

from backend.api.features.library import db as library_db
from backend.api.model import CreateGraph
from backend.api.rest_api import app as rest_app
from backend.copilot.model import ChatSession
from backend.copilot.tools.design_workflow_page import DesignWorkflowPageTool
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_graph
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

from . import db as pages_db
from . import routes

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _design(title: str, *extra_inputs: str) -> dict[str, Any]:
    fields = [{"input": name} for name in ("input_1", "input_2", *extra_inputs)]
    return {
        "title": title,
        "sections": [
            {"kind": "text", "body": "Joins two words."},
            {"kind": "run_form", "fields": fields},
            {"kind": "latest_result", "cards": []},
        ],
    }


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"workflow-page-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Workflow Page",
        }
    )
    return user.id


@pytest.fixture
async def owned(server: SpinTestServer) -> tuple[str, str]:
    """A user and the library entry of a workflow they own."""
    user_id = await _user()
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user_id
    )
    agent = await library_db.get_library_agent_by_graph_id(user_id, graph.id)
    assert agent is not None
    return user_id, agent.id


@pytest.fixture
async def client(owned: tuple[str, str]) -> AsyncIterator[tuple[AsyncClient, dict]]:
    """The routes as a client sees them, signed in as whoever ``as_user`` names."""
    app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
    app.include_router(routes.router, prefix="/api")
    as_user = {"id": owned[0]}
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": as_user["id"],
        "role": "user",
        "email": "page@example.com",
    }
    with patch.object(routes, "is_feature_enabled", AsyncMock(return_value=True)):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http:
            yield http, as_user


async def test_saving_keeps_the_design_it_replaces_and_restore_swaps_them(
    owned: tuple[str, str],
) -> None:
    user_id, library_agent_id = owned
    await pages_db.save_workflow_page(user_id, library_agent_id, _design("First"))
    saved = await pages_db.save_workflow_page(
        user_id, library_agent_id, _design("Second")
    )

    assert (saved.design["title"], saved.previous_design) == (
        "Second",
        _design("First"),
    )

    restored = await pages_db.restore_workflow_page(user_id, library_agent_id)
    assert restored.design["title"] == "First"
    assert restored.previous_design == _design("Second")


async def test_a_page_cannot_be_saved_on_another_users_workflow(
    owned: tuple[str, str],
) -> None:
    _, library_agent_id = owned
    with pytest.raises(NotFoundError):
        await pages_db.save_workflow_page(await _user(), library_agent_id, _design("X"))


async def test_the_read_route_leaves_out_a_section_that_no_longer_fits(
    owned: tuple[str, str], client: tuple[AsyncClient, dict]
) -> None:
    user_id, library_agent_id = owned
    http, _ = client
    # As if the workflow lost an input after the page was saved.
    await pages_db.save_workflow_page(
        user_id, library_agent_id, _design("Page", "input_gone")
    )

    response = await http.get(f"/api/workflow-pages/{library_agent_id}")

    assert response.status_code == 200
    page = response.json()
    assert [s["kind"] for s in page["sections"]] == ["text", "latest_result"]
    assert [(s["index"], s["kind"]) for s in page["left_out"]] == [(1, "run_form")]
    assert "input_gone" in page["left_out"][0]["problem"]


async def test_another_users_library_id_answers_404(
    owned: tuple[str, str], client: tuple[AsyncClient, dict]
) -> None:
    user_id, library_agent_id = owned
    http, as_user = client
    await pages_db.save_workflow_page(user_id, library_agent_id, _design("Mine"))

    as_user["id"] = await _user()
    response = await http.get(f"/api/workflow-pages/{library_agent_id}")

    assert response.status_code == 404


async def test_removing_a_section_keeps_the_old_design_for_restore(
    owned: tuple[str, str], client: tuple[AsyncClient, dict]
) -> None:
    user_id, library_agent_id = owned
    http, _ = client
    await pages_db.save_workflow_page(user_id, library_agent_id, _design("Page"))

    removed = await http.delete(f"/api/workflow-pages/{library_agent_id}/sections/0")
    restored = await http.post(f"/api/workflow-pages/{library_agent_id}/restore")

    assert removed.status_code == 200
    assert [s["kind"] for s in removed.json()["sections"]] == [
        "run_form",
        "latest_result",
    ]
    assert restored.json()["has_previous"] is True
    assert [s["kind"] for s in restored.json()["sections"]][0] == "text"


async def test_the_tool_previews_without_storing_and_saves_on_request(
    owned: tuple[str, str],
) -> None:
    user_id, library_agent_id = owned
    tool = DesignWorkflowPageTool()
    session = ChatSession.new(user_id=user_id, dry_run=False)

    async def call(design: dict[str, Any], save: bool) -> dict[str, Any]:
        result = await tool.execute(
            user_id,
            session,
            f"call-{uuid.uuid4().hex[:6]}",
            library_agent_id=library_agent_id,
            page=design,
            save=save,
        )
        output = result.output
        return json.loads(output) if isinstance(output, str) else output

    with patch(
        "backend.copilot.tools.design_workflow_page.is_feature_enabled",
        AsyncMock(return_value=True),
    ):
        preview = await call(_design("Preview"), save=False)
        assert preview["type"] == "workflow_page" and preview["saved"] is False
        assert await pages_db.get_workflow_page(user_id, library_agent_id) is None

        await call(_design("First"), save=True)
        second = await call(_design("Second"), save=True)

    stored = await pages_db.get_workflow_page(user_id, library_agent_id)
    assert second["saved"] is True and second["page"]["has_previous"] is True
    assert stored is not None
    assert stored.previous_design is not None
    assert (stored.design["title"], stored.previous_design["title"]) == (
        "Second",
        "First",
    )
