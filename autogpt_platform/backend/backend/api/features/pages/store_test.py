"""Pages against the database: storage, the routes, and Otto's tool."""

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
from backend.copilot.model import ChatSession, create_chat_session
from backend.copilot.tools.design_page import DesignPageTool
from backend.data.user import get_or_create_user
from backend.usecases.sample import create_test_graph
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

from . import db as pages_db
from . import routes

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _design(title: str, library_agent_id: str, *extra_inputs: str) -> dict[str, Any]:
    fields = [{"input": name} for name in ("input_1", "input_2", *extra_inputs)]
    return {
        "title": title,
        "sections": [
            {"kind": "text", "body": "Joins two words."},
            {"kind": "run_form", "workflow": library_agent_id, "fields": fields},
            {"kind": "latest_result", "workflow": library_agent_id, "cards": []},
        ],
    }


async def _user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"page-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Page",
        }
    )
    return user.id


async def _workflow(server: SpinTestServer, user_id: str) -> str:
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user_id
    )
    agent = await library_db.get_library_agent_by_graph_id(user_id, graph.id)
    assert agent is not None
    return agent.id


@pytest.fixture
async def owned(server: SpinTestServer) -> tuple[str, str]:
    """A user and the library entry of a workflow they own."""
    user_id = await _user()
    return user_id, await _workflow(server, user_id)


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


async def test_a_save_keeps_the_design_it_replaces_and_restore_swaps_them(
    owned: tuple[str, str],
) -> None:
    user_id, workflow = owned
    page = await pages_db.create_page(user_id, "First", _design("First", workflow))
    saved = await pages_db.save_page_design(
        user_id, page.id, "Second", _design("Second", workflow), based_on=1
    )

    assert (saved.revision, saved.design["title"], saved.previous_design) == (
        2,
        "Second",
        _design("First", workflow),
    )
    restored = await pages_db.restore_previous(user_id, page.id)
    assert (restored.revision, restored.design["title"]) == (3, "First")
    assert restored.previous_design == _design("Second", workflow)


async def test_a_save_based_on_an_older_revision_is_refused(
    owned: tuple[str, str]
) -> None:
    user_id, workflow = owned
    page = await pages_db.create_page(user_id, "Page", _design("Page", workflow))
    await pages_db.save_page_design(
        user_id, page.id, "B", _design("B", workflow), based_on=1
    )

    with pytest.raises(pages_db.StaleRevisionError):
        await pages_db.save_page_design(
            user_id, page.id, "C", _design("C", workflow), based_on=1
        )
    stored = await pages_db.get_page(user_id, page.id)
    assert stored is not None and stored.design["title"] == "B"


async def test_one_user_keeps_at_most_the_page_limit(owned: tuple[str, str]) -> None:
    user_id, workflow = owned
    with patch.object(pages_db, "MAX_PAGES_PER_USER", 1):
        await pages_db.create_page(user_id, "One", _design("One", workflow))
        with pytest.raises(ValueError, match="most pages"):
            await pages_db.create_page(user_id, "Two", _design("Two", workflow))


async def test_a_pin_moves_off_the_page_that_had_it(owned: tuple[str, str]) -> None:
    user_id, workflow = owned
    first = await pages_db.create_page(user_id, "A", _design("A", workflow))
    second = await pages_db.create_page(user_id, "B", _design("B", workflow))

    await pages_db.set_page_pin(user_id, first.id, workflow)
    await pages_db.set_page_pin(user_id, second.id, workflow)

    pinned = await pages_db.get_pinned_page(user_id, workflow)
    unpinned = await pages_db.get_page(user_id, first.id)
    assert pinned is not None and pinned.id == second.id
    assert unpinned is not None and unpinned.pinned_library_agent_id is None


async def test_a_page_cannot_be_pinned_to_another_users_workflow(
    server: SpinTestServer, owned: tuple[str, str]
) -> None:
    user_id, workflow = owned
    other = await _user()
    theirs = await _workflow(server, other)
    page = await pages_db.create_page(user_id, "A", _design("A", workflow))

    with pytest.raises(NotFoundError):
        await pages_db.set_page_pin(user_id, page.id, theirs)


@pytest.mark.parametrize("reference", ["another_users", "deleted", "made_up"])
async def test_a_section_on_a_workflow_the_owner_cannot_read_says_no_longer_available(
    server: SpinTestServer,
    owned: tuple[str, str],
    client: tuple[AsyncClient, dict],
    reference: str,
) -> None:
    user_id, workflow = owned
    http, _ = client
    if reference == "another_users":
        target = await _workflow(server, await _user())
    elif reference == "deleted":
        target = await _workflow(server, user_id)
        await library_db.delete_library_agent(target, user_id)
    else:
        target = str(uuid.uuid4())
    design = _design("Page", workflow)
    design["sections"].append({"kind": "recent_runs", "workflow": target})
    page = await pages_db.create_page(user_id, "Page", design)

    response = await http.get(f"/api/pages/{page.id}")

    assert response.status_code == 200
    sections = response.json()["sections"]
    assert [s["kind"] for s in sections] == [
        "text",
        "run_form",
        "latest_result",
        "notice",
    ]
    assert (
        sections[3]["reason"] == "stale"
        and "no longer available" in sections[3]["message"]
    )


async def test_a_lost_input_leaves_only_the_form_stale(
    owned: tuple[str, str], client: tuple[AsyncClient, dict]
) -> None:
    user_id, workflow = owned
    http, _ = client
    page = await pages_db.create_page(
        user_id, "Page", _design("Page", workflow, "input_gone")
    )

    sections = (await http.get(f"/api/pages/{page.id}")).json()["sections"]

    assert [(s["kind"], s.get("reason")) for s in sections] == [
        ("text", None),
        ("notice", "stale"),
        ("latest_result", None),
    ]


async def test_another_users_page_and_pin_answer_404(
    owned: tuple[str, str], client: tuple[AsyncClient, dict]
) -> None:
    user_id, workflow = owned
    http, as_user = client
    page = await pages_db.create_page(user_id, "Mine", _design("Mine", workflow))
    await pages_db.set_page_pin(user_id, page.id, workflow)
    assert (await http.get(f"/api/pages/pinned/{workflow}")).status_code == 200

    as_user["id"] = await _user()

    for method, path in [
        ("GET", f"/api/pages/{page.id}"),
        ("GET", f"/api/pages/pinned/{workflow}"),
        ("PATCH", f"/api/pages/{page.id}"),
        ("DELETE", f"/api/pages/{page.id}/sections/0"),
        ("POST", f"/api/pages/{page.id}/restore"),
        ("DELETE", f"/api/pages/{page.id}"),
    ]:
        response = await http.request(
            method, path, json={"title": "X"} if method == "PATCH" else None
        )
        assert (method, path, response.status_code) == (method, path, 404)
    assert (await http.get("/api/pages")).json() == {"pages": []}


async def test_the_edit_routes_reorder_remove_rename_restore_and_delete(
    owned: tuple[str, str], client: tuple[AsyncClient, dict]
) -> None:
    user_id, workflow = owned
    http, _ = client
    page = await pages_db.create_page(user_id, "Page", _design("Page", workflow))

    moved = await http.patch(
        f"/api/pages/{page.id}", json={"order": [2, 0, 1], "title": "Renamed"}
    )
    removed = await http.delete(f"/api/pages/{page.id}/sections/0")
    restored = await http.post(f"/api/pages/{page.id}/restore")
    listed = await http.get("/api/pages")
    deleted = await http.delete(f"/api/pages/{page.id}")

    assert [s["kind"] for s in moved.json()["sections"]] == [
        "latest_result",
        "text",
        "run_form",
    ]
    assert moved.json()["title"] == "Renamed"
    assert [s["kind"] for s in removed.json()["sections"]] == ["text", "run_form"]
    assert [s["kind"] for s in restored.json()["sections"]][0] == "latest_result"
    assert listed.json()["pages"][0]["workflow_count"] == 1
    assert deleted.status_code == 204
    assert (await http.get(f"/api/pages/{page.id}")).status_code == 404


async def test_the_tool_previews_saves_updates_reads_and_refuses_a_stale_save(
    owned: tuple[str, str],
) -> None:
    user_id, workflow = owned
    tool = DesignPageTool()
    session = await create_chat_session(user_id, dry_run=False)

    async def call(**kwargs: Any) -> dict[str, Any]:
        result = await tool.execute(
            user_id, session, f"call-{uuid.uuid4().hex[:6]}", **kwargs
        )
        output = result.output
        return json.loads(output) if isinstance(output, str) else output

    with patch(
        "backend.copilot.tools.design_page.is_feature_enabled",
        AsyncMock(return_value=True),
    ):
        catalogue = await call()
        preview = await call(page=_design("Preview", workflow))
        assert await pages_db.list_pages(user_id) == []
        created = await call(
            page=_design("First", workflow), save=True, pin_to_workflow=workflow
        )
        page_id = created["page"]["id"]
        read = await call(page_id=page_id)
        unbased = await call(
            page_id=page_id, page=_design("Second", workflow), save=True
        )
        updated = await call(
            page_id=page_id, page=_design("Second", workflow), save=True, based_on=1
        )
        again = await call(page=_design("Second page", workflow), save=True)
        stale = await call(
            page_id=page_id, page=_design("Third", workflow), save=True, based_on=1
        )
        foreign = await call(page=_design("X", str(uuid.uuid4())))

    assert "page" in catalogue["catalogue"] and catalogue["type"] == "page"
    assert preview["saved"] is False and preview["page"].get("id") is None
    assert (
        created["saved"] is True
        and created["page"]["pinned_library_agent_id"] == workflow
    )
    assert read["design"]["title"] == "First"
    assert unbased["error"] == "invalid_page"
    assert updated["page"]["revision"] == 2 and updated["page"]["has_previous"] is True
    assert stale["error"] == "stale_page"
    assert again["error"] == "stale_page" and page_id in again["message"]
    assert (
        foreign["error"] == "invalid_page"
        and "no longer available" in foreign["message"]
    )
    stored = await pages_db.get_page(user_id, page_id)
    assert stored is not None and stored.chat_session_id == session.session_id


async def test_the_tool_says_unavailable_without_the_flag(
    owned: tuple[str, str]
) -> None:
    user_id, workflow = owned
    session = ChatSession.new(user_id=user_id, dry_run=False)
    with patch(
        "backend.copilot.tools.design_page.is_feature_enabled",
        AsyncMock(return_value=False),
    ):
        result = await DesignPageTool().execute(
            user_id, session, "call-off", page=_design("X", workflow), save=True
        )
    output = (
        json.loads(result.output) if isinstance(result.output, str) else result.output
    )
    assert output["error"] == "unavailable"
    assert await pages_db.list_pages(user_id) == []
