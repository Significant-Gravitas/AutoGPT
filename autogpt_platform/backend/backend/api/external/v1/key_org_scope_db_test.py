"""A v1 key reads and starts runs only of workflows in the organization it names.

Against Postgres, through the real External API app: real keys, organizations, a
workflow and a run. Only the rate limiter's counter, the paywall and the
executor's queue are stubbed.
"""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from prisma.enums import AgentExecutionStatus, APIKeyPermission
from prisma.models import (
    AgentGraphExecution,
    Organization,
    OrgMember,
    Team,
    TeamMember,
    User,
)
from pydantic import BaseModel

from backend.api.external import middleware
from backend.api.external.fastapi_app import external_api
from backend.api.external.v1 import routes as v1_routes
from backend.data import db
from backend.data import graph as graph_db
from backend.data.auth import api_key


class Setup(BaseModel):
    """A user in two organizations, with a workflow and a run in ``home``, and a
    teammate's workflow and run in ``elsewhere``, visible to all its members."""

    user: str
    home: str
    elsewhere: str
    graph_id: str
    run_id: str
    teammate_graph_id: str
    teammate_run_id: str
    keys: dict[str, str]


@pytest_asyncio.fixture(loop_scope="session")
async def setup() -> AsyncIterator[Setup]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user, teammate = (f"key-org-scope-{uuid.uuid4()}" for _ in range(2))
    for user_id in (user, teammate):
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com"}
        )
    home = await _org("Home", user, personal=True)
    elsewhere = await _org("Elsewhere", user)
    await OrgMember.prisma().create(
        data={"orgId": elsewhere, "userId": teammate, "status": "ACTIVE"}
    )
    graph, run = await _workflow_with_run(user, home)
    teammate_graph, teammate_run = await _workflow_with_run(teammate, elsewhere)
    keys = {
        "home": await _key(user, home),
        "elsewhere": await _key(user, elsewhere),
        "no organization": await _key(user, None),
    }
    try:
        yield Setup(
            user=user,
            home=home,
            elsewhere=elsewhere,
            graph_id=graph.id,
            run_id=run.id,
            teammate_graph_id=teammate_graph.id,
            teammate_run_id=teammate_run.id,
            keys=keys,
        )
    finally:
        await User.prisma().delete_many(where={"id": {"in": [user, teammate]}})
        await Organization.prisma().delete_many(where={"id": {"in": [home, elsewhere]}})
        if owned_connection:
            await db.disconnect()


@pytest.fixture(autouse=True)
def fresh_membership_answers():
    middleware.holds_membership.cache_clear()
    yield
    middleware.holds_membership.cache_clear()


@pytest.fixture
def queue(mocker) -> AsyncMock:
    mocker.patch.object(v1_routes, "enforce_payment_paywall", AsyncMock())
    return mocker.patch.object(
        v1_routes, "add_graph_execution", AsyncMock(return_value=MagicMock(id="run"))
    )


@pytest_asyncio.fixture(loop_scope="session")
async def client(v1_counter) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=external_api), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.parametrize(
    "key, status",
    [("home", 200), ("elsewhere", 404), ("no organization", 200)],
)
async def test_run_results_are_found_only_through_a_key_for_their_organization(
    setup: Setup, client: AsyncClient, key: str, status: int
):
    response = await client.get(
        f"/v1/graphs/{setup.graph_id}/executions/{setup.run_id}/results",
        headers={"X-API-Key": setup.keys[key]},
    )

    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["execution_id"] == setup.run_id


async def test_run_results_are_not_found_under_another_workflow_id(
    setup: Setup, client: AsyncClient
):
    response = await client.get(
        f"/v1/graphs/{uuid.uuid4()}/executions/{setup.run_id}/results",
        headers={"X-API-Key": setup.keys["home"]},
    )

    assert response.status_code == 404


async def test_a_key_does_not_read_a_teammates_run_in_its_organization(
    setup: Setup, client: AsyncClient
):
    response = await client.get(
        f"/v1/graphs/{setup.teammate_graph_id}/executions/"
        f"{setup.teammate_run_id}/results",
        headers={"X-API-Key": setup.keys["elsewhere"]},
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "key, status",
    [("home", 200), ("elsewhere", 404), ("no organization", 200)],
)
async def test_a_workflow_is_run_only_through_a_key_for_its_organization(
    setup: Setup, client: AsyncClient, queue: AsyncMock, key: str, status: int
):
    response = await client.post(
        f"/v1/graphs/{setup.graph_id}/execute/1",
        headers={"X-API-Key": setup.keys[key]},
        json={"node_input": {}},
    )

    assert response.status_code == status, response.text
    assert queue.await_count == (1 if status == 200 else 0)


async def _workflow_with_run(
    user_id: str, organization_id: str
) -> tuple[graph_db.GraphModel, AgentGraphExecution]:
    graph = await graph_db.create_graph(
        graph_db.Graph(name="Weekly report", description=""),
        user_id,
        organization_id=organization_id,
    )
    run = await AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "userId": user_id,
            "organizationId": organization_id,
            "executionStatus": AgentExecutionStatus.COMPLETED,
        }
    )
    return graph, run


async def _org(name: str, owner: str, *, personal: bool = False) -> str:
    org = await Organization.prisma().create(
        data={
            "name": name,
            "slug": f"key-org-scope-{uuid.uuid4().hex[:8]}",
            "isPersonal": personal,
        }
    )
    home = await Team.prisma().create(
        data={"name": "General", "orgId": org.id, "isDefault": True}
    )
    await OrgMember.prisma().create(
        data={
            "orgId": org.id,
            "userId": owner,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )
    await TeamMember.prisma().create(
        data={"teamId": home.id, "userId": owner, "isAdmin": True, "status": "ACTIVE"}
    )
    return org.id


async def _key(user_id: str, organization_id: str | None) -> str:
    _, plaintext = await api_key.create_api_key(
        name="org scope",
        user_id=user_id,
        permissions=[APIKeyPermission.READ_GRAPH, APIKeyPermission.EXECUTE_GRAPH],
        organization_id=organization_id,
    )
    return plaintext
