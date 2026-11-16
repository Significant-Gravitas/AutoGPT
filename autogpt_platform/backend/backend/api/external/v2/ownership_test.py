"""
A member sees a colleague's org-home runs and graphs through v2, but the writes
beneath these routes are the owner's, so each refuses a colleague before acting.

These drive the real data layer: the defect sits between an org-visible read and
an owner-scoped write, and a mocked database absorbs it.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import fastapi
import httpx
import prisma.models
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission
from pydantic import BaseModel

from backend.data import execution as execution_db
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

from . import graphs
from .errors import add_v2_exception_handlers
from .models import GraphCreateRequest
from .runs import runs_router
from .tenancy import TenantContext, require_auth


class Colleagues(BaseModel):
    owner: TenantContext
    mate: TenantContext
    graph_id: str
    run_id: str


@pytest.fixture
async def colleagues(server: SpinTestServer) -> Colleagues:
    org_id = str(uuid4())
    owner, mate = [await _member(org_id) for _ in range(2)]
    graph = await graphs.create_graph(
        GraphCreateRequest(name="Shared", nodes=[], links=[]), auth=owner
    )
    run = await execution_db.create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={},
        user_id=owner.user_id,
        organization_id=org_id,
    )
    return Colleagues(owner=owner, mate=mate, graph_id=graph.id, run_id=run.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_colleagues_run_is_not_deleted(colleagues: Colleagues) -> None:
    refused = await _call("DELETE", f"/runs/{colleagues.run_id}", colleagues.mate)

    assert refused.status_code == 403, refused.text
    assert not (await _run_row(colleagues.run_id)).isDeleted

    deleted = await _call("DELETE", f"/runs/{colleagues.run_id}", colleagues.owner)

    assert deleted.status_code == 204, deleted.text
    assert (await _run_row(colleagues.run_id)).isDeleted


@pytest.mark.asyncio(loop_scope="session")
async def test_a_colleagues_run_is_not_cancelled(
    colleagues: Colleagues, mocker: pytest_mock.MockFixture
) -> None:
    queue = Mock(publish_message=AsyncMock())
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue",
        new_callable=AsyncMock,
        return_value=queue,
    )

    refused = await _call("POST", f"/runs/{colleagues.run_id}/stop", colleagues.mate)

    assert refused.status_code == 403, refused.text
    queue.publish_message.assert_not_awaited()

    stopped = await _call("POST", f"/runs/{colleagues.run_id}/stop", colleagues.owner)

    assert stopped.status_code == 202, stopped.text
    queue.publish_message.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_colleague_cannot_append_a_version(colleagues: Colleagues) -> None:
    body = {"name": "Shared", "nodes": [], "links": []}

    refused = await _call(
        "PUT", f"/graphs/{colleagues.graph_id}", colleagues.mate, json=body
    )

    assert refused.status_code == 403, refused.text
    assert await _version_owners(colleagues.graph_id) == {1: colleagues.owner.user_id}

    updated = await _call(
        "PUT", f"/graphs/{colleagues.graph_id}", colleagues.owner, json=body
    )

    assert updated.status_code == 200, updated.text
    assert await _version_owners(colleagues.graph_id) == {
        1: colleagues.owner.user_id,
        2: colleagues.owner.user_id,
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_a_colleague_cannot_activate_a_version(
    colleagues: Colleagues, mocker: pytest_mock.MockFixture
) -> None:
    activate = mocker.spy(graphs, "before_graph_activate")
    path = f"/graphs/{colleagues.graph_id}/versions/active"
    body = {"active_graph_version": 1}

    refused = await _call("PUT", path, colleagues.mate, json=body)

    assert refused.status_code == 403, refused.text
    activate.assert_not_called()

    activated = await _call("PUT", path, colleagues.owner, json=body)

    assert activated.status_code == 204, activated.text
    activate.assert_called_once()


async def _member(org_id: str) -> TenantContext:
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    return TenantContext(
        user_id=user_id,
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id=org_id,
    )


async def _call(
    method: str, path: str, auth: TenantContext, json: dict | None = None
) -> httpx.Response:
    app = fastapi.FastAPI()
    app.include_router(runs_router, prefix="/runs")
    app.include_router(graphs.graphs_router, prefix="/graphs")
    add_v2_exception_handlers(app)
    app.dependency_overrides[require_auth] = lambda: auth
    # Without raise_app_exceptions=False a 500 surfaces as the exception, not a response.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://v2") as client:
        return await client.request(method, path, json=json)


async def _run_row(run_id: str) -> prisma.models.AgentGraphExecution:
    return await prisma.models.AgentGraphExecution.prisma().find_unique_or_raise(
        where={"id": run_id}
    )


async def _version_owners(graph_id: str) -> dict[int, str]:
    versions = await prisma.models.AgentGraph.prisma().find_many(where={"id": graph_id})
    return {v.version: v.userId for v in versions}
