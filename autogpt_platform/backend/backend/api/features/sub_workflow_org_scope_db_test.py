"""A sub-workflow runs only if it belongs to its parent run's organization, or is
a marketplace workflow: end to end, a parent run started through the real route
in the user's work organization, then its sub-workflow node run by the real
block with the context the executor hands it. Only the executor's queue and the
wait on the sub-run's events are stubbed.

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
from backend.api.features.library import db as library_db
from backend.api.utils.exceptions import add_exception_handlers
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import StoreValueBlock
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data import graph as graph_db
from backend.data.db import prisma
from backend.data.execution import GraphExecutionEntry
from backend.data.org_migration import create_personal_org
from backend.util.exceptions import GraphNotAccessibleError

USER_HEADER = "X-Test-User"


class World(BaseModel):
    """A user with a personal org and a work org, and a work-org workflow for
    each sub-workflow, which runs it as its one node."""

    user: str
    work: str
    parents: dict[str, str]
    subs: dict[str, str]


@pytest_asyncio.fixture(loop_scope="session")
async def world() -> AsyncIterator[World]:
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    user = f"sub-workflow-org-{uuid.uuid4().hex[:8]}"
    creator = f"sub-workflow-creator-{uuid.uuid4().hex[:8]}"
    orgs: list[str] = []
    try:
        for user_id in (user, creator):
            await prisma.user.create(
                data={"id": user_id, "email": f"{user_id}@example.com"}
            )
            orgs.append((await create_personal_org(user_id, user_id, user_id)).id)
        personal = orgs[0]
        work = await prisma.organization.create(
            data={"name": "Work", "slug": f"work-{uuid.uuid4().hex[:8]}"}
        )
        orgs.append(work.id)
        work_team = await prisma.team.create(
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
        subs = {
            "work workflow": await _workflow(user, work.id),
            "personal workflow": await _workflow(user, personal),
            "marketplace workflow": await _workflow(creator, orgs[1]),
        }
        await _publish(creator, subs["marketplace workflow"])
        parents = {}
        for kind, sub in subs.items():
            parent, _ = await library_db.create_graph_in_library(
                graph_db.Graph(
                    name="Parent", description="", nodes=[_sub_workflow_node(sub)]
                ),
                user,
                organization_id=work.id,
                team_id=work_team.id,
            )
            parents[kind] = parent.id
        yield World(
            user=user,
            work=work.id,
            parents=parents,
            subs={kind: sub.id for kind, sub in subs.items()},
        )
    finally:
        try:
            if prisma.is_connected():
                users = {"in": [user, creator]}
                await prisma.storelisting.delete_many(where={"owningUserId": users})
                await prisma.libraryagent.delete_many(where={"userId": users})
                await prisma.user.delete_many(where={"id": users})
                await prisma.organization.delete_many(where={"id": {"in": orgs}})
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


@pytest_asyncio.fixture(loop_scope="session")
async def api(world: World) -> AsyncIterator[httpx.AsyncClient]:
    app = fastapi.FastAPI()
    app.include_router(graphs_router)
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
def queue(mocker) -> AsyncMock:
    """The executor's queue: what is published is what the executor would run."""
    publish = AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue",
        AsyncMock(return_value=MagicMock(publish_message=publish)),
    )
    mocker.patch(
        "backend.executor.utils.is_user_paywalled", AsyncMock(return_value=False)
    )
    mocker.patch(
        "backend.executor.utils.is_user_refused", AsyncMock(return_value=False)
    )
    # The block waits on the sub-run's events, which only a running executor sends.
    mocker.patch.object(AgentExecutorBlock, "_run", _no_outputs)
    return publish


@pytest.mark.parametrize(
    "sub, runs",
    [
        ("work workflow", True),
        ("personal workflow", False),
        ("marketplace workflow", True),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_sub_workflow_runs_only_from_its_organization_or_the_marketplace(
    api: httpx.AsyncClient, world: World, queue: AsyncMock, sub: str, runs: bool
):
    response = await api.post(
        f"/graphs/{world.parents[sub]}/execute/1",
        headers={USER_HEADER: world.user, "X-Org-Id": world.work},
        json={"inputs": {}, "dry_run": True},
    )
    assert response.status_code == 200, response.text
    parent_run = GraphExecutionEntry.model_validate_json(
        queue.await_args.kwargs["message"]
    )
    node = (await graph_db.get_graph(world.parents[sub], 1, world.user)).nodes[0]
    # As the executor shapes a sub-workflow node's input (manager.py).
    input_data = AgentExecutorBlock.Input(**node.input_default)
    input_data.user_id = world.user

    start = _drain(
        AgentExecutorBlock().run(
            input_data,
            graph_exec_id=parent_run.graph_exec_id,
            execution_context=parent_run.execution_context,
        )
    )
    if runs:
        await start
    else:
        with pytest.raises(GraphNotAccessibleError):
            await start

    sub_runs = await prisma.agentgraphexecution.find_many(
        where={"agentGraphId": world.subs[sub], "userId": world.user}
    )
    assert [r.organizationId for r in sub_runs] == ([world.work] if runs else [])


async def _drain(outputs) -> None:
    async for _ in outputs:
        pass


async def _no_outputs(self, **kwargs):
    return
    yield


async def _workflow(user_id: str, org_id: str) -> graph_db.GraphModel:
    return await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(
                name="Briefing",
                description="",
                nodes=[
                    graph_db.Node(
                        block_id=StoreValueBlock().id, input_default={"input": "x"}
                    )
                ],
            ),
            user_id,
        ),
        user_id,
        organization_id=org_id,
    )


def _sub_workflow_node(sub: graph_db.GraphModel) -> graph_db.Node:
    return graph_db.Node(
        block_id=AgentExecutorBlock().id,
        input_default={
            "user_id": sub.user_id,
            "graph_id": sub.id,
            "graph_version": sub.version,
            "inputs": {},
            "input_schema": {},
            "output_schema": {},
        },
    )


async def _publish(user_id: str, graph: graph_db.GraphModel) -> None:
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
            "agentGraphId": graph.id,
            "owningUserId": user_id,
        }
    )
    await prisma.storelistingversion.create(
        data={
            "storeListingId": listing.id,
            "version": 1,
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "name": "Briefing",
            "subHeading": "",
            "description": "",
            "imageUrls": [],
            "categories": [],
            "submissionStatus": SubmissionStatus.APPROVED,
        }
    )
