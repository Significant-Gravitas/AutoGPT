"""A v1 key reads and starts runs only of workflows in the organization it names.

Against Postgres, through the real External API app: real keys, organizations,
workflows, runs and a marketplace submission. Only the rate limiter's counter, the paywall and the
executor's queue are stubbed.
"""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from prisma.enums import AgentExecutionStatus, APIKeyPermission, SubmissionStatus
from prisma.models import (
    AgentGraphExecution,
    LibraryAgent,
    Organization,
    OrgMember,
    Profile,
    StoreListing,
    StoreListingVersion,
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
    """A user in two organizations. In ``home`` they keep a workflow, one they
    submitted to the marketplace (so also in their library), and a run of each and
    of an untagged workflow from before organizations; one run of their ``home``
    workflow was recorded in ``elsewhere``. In ``elsewhere`` they keep a workflow
    and a run of it too, and a teammate keeps a workflow and a run, visible to all
    its members."""

    user: str
    home: str
    elsewhere: str
    runs: dict[str, tuple[str, str]]
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
    runs = {
        "workflow": await _workflow_with_run(user, home, home),
        "untagged workflow": await _workflow_with_run(user, None, home),
        "submitted workflow": await _workflow_with_run(user, home, home),
        "run recorded elsewhere": await _workflow_with_run(user, home, elsewhere),
        "workflow elsewhere": await _workflow_with_run(user, elsewhere, elsewhere),
        "teammate's workflow": await _workflow_with_run(teammate, elsewhere, elsewhere),
    }
    await _submit_to_marketplace(user, runs["submitted workflow"][0], home)
    keys = {
        "home": await _key(user, home),
        "elsewhere": await _key(user, elsewhere),
        "no organization": await _key(user, None),
    }
    try:
        yield Setup(user=user, home=home, elsewhere=elsewhere, runs=runs, keys=keys)
    finally:
        users = {"in": [user, teammate]}
        await StoreListing.prisma().delete_many(where={"owningUserId": users})
        await LibraryAgent.prisma().delete_many(where={"userId": users})
        await User.prisma().delete_many(where={"id": users})
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


KEYS_AND_STATUSES = [("home", 200), ("elsewhere", 404), ("no organization", 200)]
# A key that names no organization acts in its owner's personal one, ``home``.
ELSEWHERE_KEYS_AND_STATUSES = [
    ("home", 404),
    ("elsewhere", 200),
    ("no organization", 404),
]


@pytest.mark.parametrize("key, status", KEYS_AND_STATUSES)
@pytest.mark.parametrize("workflow", ["workflow", "untagged workflow"])
async def test_run_results_are_found_only_through_a_key_for_their_organization(
    setup: Setup, client: AsyncClient, workflow: str, key: str, status: int
):
    graph_id, run_id = setup.runs[workflow]
    response = await client.get(
        f"/v1/graphs/{graph_id}/executions/{run_id}/results",
        headers={"X-API-Key": setup.keys[key]},
    )

    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["execution_id"] == run_id


@pytest.mark.parametrize("key, status", ELSEWHERE_KEYS_AND_STATUSES)
async def test_a_workflow_in_another_organization_is_reached_only_through_its_key(
    setup: Setup, client: AsyncClient, queue: AsyncMock, key: str, status: int
):
    graph_id, run_id = setup.runs["workflow elsewhere"]
    headers = {"X-API-Key": setup.keys[key]}

    results = await client.get(
        f"/v1/graphs/{graph_id}/executions/{run_id}/results", headers=headers
    )
    started = await client.post(
        f"/v1/graphs/{graph_id}/execute/1", headers=headers, json={"node_input": {}}
    )

    assert (results.status_code, started.status_code) == (status, status)
    assert queue.await_count == (1 if status == 200 else 0)


async def test_run_results_are_not_found_under_another_workflow_id(
    setup: Setup, client: AsyncClient
):
    _, run_id = setup.runs["workflow"]
    response = await client.get(
        f"/v1/graphs/{uuid.uuid4()}/executions/{run_id}/results",
        headers={"X-API-Key": setup.keys["home"]},
    )

    assert response.status_code == 404


async def test_a_run_is_not_read_in_its_organization_if_its_workflow_is_elsewhere(
    setup: Setup, client: AsyncClient
):
    graph_id, run_id = setup.runs["run recorded elsewhere"]
    response = await client.get(
        f"/v1/graphs/{graph_id}/executions/{run_id}/results",
        headers={"X-API-Key": setup.keys["elsewhere"]},
    )

    assert response.status_code == 404


async def test_a_key_does_not_read_a_teammates_run_in_its_organization(
    setup: Setup, client: AsyncClient
):
    graph_id, run_id = setup.runs["teammate's workflow"]
    response = await client.get(
        f"/v1/graphs/{graph_id}/executions/{run_id}/results",
        headers={"X-API-Key": setup.keys["elsewhere"]},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("key, status", KEYS_AND_STATUSES)
@pytest.mark.parametrize("workflow", ["workflow", "submitted workflow"])
async def test_a_workflow_is_run_only_through_a_key_for_its_organization(
    setup: Setup,
    client: AsyncClient,
    queue: AsyncMock,
    workflow: str,
    key: str,
    status: int,
):
    graph_id, _ = setup.runs[workflow]
    response = await client.post(
        f"/v1/graphs/{graph_id}/execute/1",
        headers={"X-API-Key": setup.keys[key]},
        json={"node_input": {}},
    )

    assert response.status_code == status, response.text
    assert queue.await_count == (1 if status == 200 else 0)


async def _workflow_with_run(
    user_id: str, graph_org: str | None, run_org: str
) -> tuple[str, str]:
    graph = await graph_db.create_graph(
        graph_db.Graph(name="Weekly report", description=""),
        user_id,
        organization_id=graph_org,
    )
    run = await AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "userId": user_id,
            "organizationId": run_org,
            "executionStatus": AgentExecutionStatus.COMPLETED,
        }
    )
    return graph.id, run.id


async def _submit_to_marketplace(user_id: str, graph_id: str, org_id: str) -> None:
    """A pending submission, and the library entry its creator keeps."""
    await Profile.prisma().create(
        data={
            "userId": user_id,
            "name": "Creator",
            "username": f"creator-{uuid.uuid4().hex[:8]}",
            "description": "",
            "links": [],
        }
    )
    listing = await StoreListing.prisma().create(
        data={
            "slug": f"weekly-report-{uuid.uuid4().hex[:8]}",
            "agentGraphId": graph_id,
            "owningUserId": user_id,
        }
    )
    await StoreListingVersion.prisma().create(
        data={
            "storeListingId": listing.id,
            "version": 1,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "name": "Weekly report",
            "subHeading": "",
            "description": "",
            "imageUrls": [],
            "categories": [],
            "submissionStatus": SubmissionStatus.PENDING,
        }
    )
    await LibraryAgent.prisma().create(
        data={
            "userId": user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "organizationId": org_id,
        }
    )


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
