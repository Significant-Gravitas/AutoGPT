"""A workflow is read, run and scheduled only in an organization it belongs to,
end to end: the real routes, the real request-context resolution, and membership
rows in Postgres. Only the executor's queue and the scheduler service are stubbed.

Lives outside ``orgs/``, whose conftest stubs the ``server`` fixture; these
tests need only Postgres, so they connect Prisma themselves.
"""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth.dependencies import get_unbanned_jwt_payload
from prisma.enums import SubmissionStatus
from pydantic import BaseModel

from backend.api.features.graphs.routes import router as graphs_router
from backend.api.features.schedules.routes import router as schedules_router
from backend.api.utils.exceptions import add_exception_handlers
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data import graph as graph_db
from backend.data.db import prisma
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.org_migration import create_personal_org
from backend.executor.scheduler import GraphExecutionJobInfo

USER_HEADER = "X-Test-User"


class World(BaseModel):
    """A user who keeps two workflows in their personal org, one of them submitted
    to the marketplace and so also in their library, and who also owns ``work``."""

    user: str
    personal: str
    work: str
    graphs: dict[str, str]


@pytest_asyncio.fixture(loop_scope="session")
async def world() -> AsyncIterator[World]:
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    user = f"workflow-org-scope-{uuid.uuid4().hex[:8]}"
    orgs: list[str] = []
    try:
        await prisma.user.create(data={"id": user, "email": f"{user}@example.com"})
        personal = await create_personal_org(user, user, user)
        orgs.append(personal.id)
        work = await prisma.organization.create(
            data={"name": "Work", "slug": f"work-{uuid.uuid4().hex[:8]}"}
        )
        orgs.append(work.id)
        await prisma.team.create(
            data={"name": "General", "orgId": work.id, "isDefault": True}
        )
        await prisma.orgmember.create(
            data={
                "orgId": work.id,
                "userId": user,
                "status": "ACTIVE",
                "isOwner": True,
                "isAdmin": True,
            }
        )
        graphs = {
            kind: (await _workflow(user, personal.id)).id
            for kind in ("workflow", "submitted workflow")
        }
        await _submit_to_marketplace(user, graphs["submitted workflow"], personal.id)
        yield World(user=user, personal=personal.id, work=work.id, graphs=graphs)
    finally:
        try:
            if prisma.is_connected():
                await prisma.storelisting.delete_many(where={"owningUserId": user})
                await prisma.libraryagent.delete_many(where={"userId": user})
                await prisma.user.delete_many(where={"id": user})
                await prisma.organization.delete_many(where={"id": {"in": orgs}})
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


@pytest_asyncio.fixture(loop_scope="session")
async def api(world: World) -> AsyncIterator[httpx.AsyncClient]:
    app = fastapi.FastAPI()
    app.include_router(graphs_router)
    app.include_router(schedules_router)
    add_exception_handlers(app)

    async def token(request: fastapi.Request) -> dict:
        user_id = request.headers[USER_HEADER]
        return {"sub": user_id, "role": "authenticated", "email": f"{user_id}@x.io"}

    app.dependency_overrides[get_unbanned_jwt_payload] = token
    app.dependency_overrides[enforce_payment_paywall] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def queue(mocker, world: World) -> AsyncMock:
    run = GraphExecutionMeta(
        id=str(uuid.uuid4()),
        user_id=world.user,
        graph_id=world.graphs["workflow"],
        graph_version=1,
        inputs={},
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=ExecutionStatus.QUEUED,
        stats=None,
    )
    return mocker.patch(
        "backend.api.features.graphs.routes.execution_utils.add_graph_execution",
        AsyncMock(return_value=run),
    )


@pytest.fixture
def scheduler(mocker, world: World) -> AsyncMock:
    job = GraphExecutionJobInfo(
        id=str(uuid.uuid4()),
        name="Weekly",
        next_run_time="",
        user_id=world.user,
        graph_id=world.graphs["workflow"],
        graph_version=1,
        cron="0 9 * * 1",
        input_data={},
    )
    client = MagicMock()
    client.add_execution_schedule = AsyncMock(return_value=job)
    mocker.patch(
        "backend.api.features.schedules.routes.get_scheduler_client",
        return_value=client,
    )
    mocker.patch(
        "backend.api.features.schedules.routes.complete_onboarding_step", AsyncMock()
    )
    return client.add_execution_schedule


ORGS_AND_STATUSES = [("personal", 200), ("work", 404)]
WORKFLOWS = ["workflow", "submitted workflow"]


@pytest.mark.parametrize("org, status", ORGS_AND_STATUSES)
@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_workflow_is_read_only_in_an_organization_it_belongs_to(
    api: httpx.AsyncClient, world: World, workflow: str, org: str, status: int
):
    response = await api.get(
        f"/graphs/{world.graphs[workflow]}",
        headers={USER_HEADER: world.user, "X-Org-Id": getattr(world, org)},
    )

    assert response.status_code == status, response.text


@pytest.mark.parametrize("org, status", ORGS_AND_STATUSES)
@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_workflow_runs_only_in_an_organization_it_belongs_to(
    api: httpx.AsyncClient,
    world: World,
    queue: AsyncMock,
    workflow: str,
    org: str,
    status: int,
):
    response = await api.post(
        f"/graphs/{world.graphs[workflow]}/execute/1",
        headers={USER_HEADER: world.user, "X-Org-Id": getattr(world, org)},
        json={"inputs": {}, "dry_run": True},
    )

    assert response.status_code == status, response.text
    assert queue.await_count == (1 if status == 200 else 0)


@pytest.mark.parametrize("org, status", ORGS_AND_STATUSES)
@pytest.mark.parametrize("workflow", WORKFLOWS)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_workflow_is_scheduled_only_in_an_organization_it_belongs_to(
    api: httpx.AsyncClient,
    world: World,
    scheduler: AsyncMock,
    workflow: str,
    org: str,
    status: int,
):
    response = await api.post(
        f"/graphs/{world.graphs[workflow]}/schedules",
        headers={USER_HEADER: world.user, "X-Org-Id": getattr(world, org)},
        json={"name": "Weekly", "cron": "0 9 * * 1", "inputs": {}, "timezone": "UTC"},
    )

    assert response.status_code == status, response.text
    assert scheduler.await_count == (1 if status == 200 else 0)


async def _workflow(user_id: str, org_id: str) -> graph_db.GraphModel:
    return await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(name="Briefing", description=""), user_id
        ),
        user_id,
        organization_id=org_id,
    )


async def _submit_to_marketplace(user_id: str, graph_id: str, org_id: str) -> None:
    """A pending submission, and the library entry its creator keeps."""
    await prisma.profile.create(
        data={
            "userId": user_id,
            "name": "Creator",
            "username": f"creator-{uuid.uuid4().hex[:8]}",
            "description": "",
            "links": [],
        }
    )
    listing = await prisma.storelisting.create(
        data={
            "slug": f"briefing-{uuid.uuid4().hex[:8]}",
            "agentGraphId": graph_id,
            "owningUserId": user_id,
        }
    )
    await prisma.storelistingversion.create(
        data={
            "storeListingId": listing.id,
            "version": 1,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "name": "Briefing",
            "subHeading": "",
            "description": "",
            "imageUrls": [],
            "categories": [],
            "submissionStatus": SubmissionStatus.PENDING,
        }
    )
    await prisma.libraryagent.create(
        data={
            "userId": user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "organizationId": org_id,
        }
    )
