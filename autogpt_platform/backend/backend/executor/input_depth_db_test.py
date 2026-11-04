import json
from unittest.mock import AsyncMock
from uuid import uuid4

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from prisma.errors import MissingRequiredValueError

from backend.api.features.graph_executions.routes import router as executions_router
from backend.api.features.graphs.routes import router as graphs_router
from backend.api.utils.exceptions import add_exception_handlers
from backend.blocks.data_manipulation import CreateListBlock
from backend.blocks.io import AgentInputBlock
from backend.data.db import prisma
from backend.data.execution import create_graph_execution, get_graph_execution
from backend.data.org_migration import ensure_personal_org


@pytest_asyncio.fixture(loop_scope="session")
async def owner():
    connected_here = not prisma.is_connected()
    user_id = str(uuid4())
    try:
        if connected_here:
            await prisma.connect()
        await prisma.user.create(
            data={"id": user_id, "email": f"depth-{user_id}@example.com"}
        )
        await ensure_personal_org(user_id)
        for block in (AgentInputBlock(), CreateListBlock()):
            await prisma.agentblock.upsert(
                where={"id": block.id},
                data={
                    "create": {
                        "id": block.id,
                        "name": block.name,
                        "inputSchema": json.dumps(block.input_schema.jsonschema()),
                        "outputSchema": json.dumps(block.output_schema.jsonschema()),
                    },
                    "update": {},
                },
            )
        yield user_id
    finally:
        try:
            if prisma.is_connected():
                # The user's graphs and runs go with them, before the
                # organization whose team holds them.
                await prisma.user.delete_many(where={"id": user_id})
                await prisma.organization.delete_many(
                    where={"bootstrapUserId": user_id}
                )
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


@pytest_asyncio.fixture(loop_scope="session")
async def personal_team(owner) -> tuple[str, str]:
    """The owner's personal organization and its default team, where a
    personal workflow and its runs live."""
    org = await prisma.organization.find_first_or_raise(
        where={"bootstrapUserId": owner, "isPersonal": True}
    )
    team = await prisma.team.find_first_or_raise(
        where={"orgId": org.id, "isDefault": True}
    )
    return org.id, team.id


@pytest_asyncio.fixture(loop_scope="session")
async def api(owner, personal_team):
    org_id, team_id = personal_team
    app = fastapi.FastAPI()
    app.include_router(graphs_router)
    app.include_router(executions_router)
    add_exception_handlers(app)

    async def user_id():
        return owner

    async def context():
        return RequestContext(
            user_id=owner,
            org_id=org_id,
            team_id=team_id,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_user_id] = user_id
    app.dependency_overrides[requires_user] = user_id
    app.dependency_overrides[get_request_context] = context
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def execution_queue(mocker) -> AsyncMock:
    """The run queue, held here: a live executor (the server fixture's, once a
    test in the session has started it) would pick up an accepted run and hold
    a live lease on the owner's team while this test tears the team down."""
    queue = AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue",
        AsyncMock(return_value=queue),
    )
    return queue


def nested(depth):
    value = "abyss"
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "column,depth", [("inputs", 126), ("mask", 125), ("node", 127)]
)
async def test_prisma_roundtrip_at_stored_json_limit(
    owner, personal_team, column, depth
):
    org_id, team_id = personal_team
    graph = await prisma.agentgraph.create(
        data={
            "userId": owner,
            "organizationId": org_id,
            "teamId": team_id,
            "Nodes": {"create": [{"agentBlockId": CreateListBlock().id}]},
        },
        include={"Nodes": True},
    )
    # A run is written only for a workflow in its owner's library.
    await prisma.libraryagent.create(
        data={
            "userId": owner,
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "isCreatedByUser": True,
            "organizationId": org_id,
            "teamId": team_id,
        }
    )
    node_id = graph.Nodes[0].id

    async def write(value):
        return await create_graph_execution(
            graph_id=graph.id,
            graph_version=graph.version,
            user_id=owner,
            inputs={"topic": value} if column == "inputs" else {},
            nodes_input_masks={node_id: {"values": value}} if column == "mask" else {},
            starting_nodes_input=(
                [(node_id, {"values": value})] if column == "node" else []
            ),
            organization_id=org_id,
            team_id=team_id,
        )

    value = nested(depth)
    run = await write(value)
    loaded = await get_graph_execution(
        owner,
        run.id,
        include_node_executions=True,
        organization_id=org_id,
        team_id_restriction=team_id,
    )
    assert loaded is not None
    if column == "inputs":
        assert loaded.inputs == {"topic": value}
    elif column == "mask":
        assert loaded.nodes_input_masks == {node_id: {"values": value}}
    else:
        assert loaded.node_executions[0].input_data["values"] == value

    with pytest.raises(MissingRequiredValueError, match="recursion limit exceeded"):
        await write(nested(depth + 1))
    assert await prisma.agentgraphexecution.count(where={"userId": owner}) == 1


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("source", ["raw_input", "encoded_default"])
async def test_rejected_run_keeps_both_execution_lists_readable(
    api, owner, execution_queue, source
):
    async def create_workflow(depth):
        block = AgentInputBlock() if source == "raw_input" else CreateListBlock()
        defaults = (
            {"name": "topic"}
            if source == "raw_input"
            else {"values": json.dumps(nested(depth))}
        )
        response = await api.post(
            "/graphs",
            json={
                "graph": {
                    "name": "Input depth regression",
                    "description": "Execution storage boundary regression",
                    "nodes": [{"block_id": block.id, "input_default": defaults}],
                    "links": [],
                }
            },
        )
        assert response.status_code == 200, response.text
        return response.json()["id"]

    depth = 126 if source == "raw_input" else 127
    graph_id = await create_workflow(depth)
    inputs = {"topic": nested(depth)} if source == "raw_input" else {}
    accepted = await api.post(f"/graphs/{graph_id}/execute/1", json={"inputs": inputs})
    assert accepted.status_code == 200, accepted.text
    run_id = accepted.json()["id"]
    assert accepted.json()["inputs"] == inputs
    assert accepted.json()["status"] == "QUEUED"

    rejected_graph = graph_id
    if source == "encoded_default":
        rejected_graph = await create_workflow(depth + 1)
    rejected = await api.post(
        f"/graphs/{rejected_graph}/execute/1",
        json={"inputs": {"topic": nested(depth + 1)} if source == "raw_input" else {}},
    )
    assert rejected.status_code == 400, rejected.text
    assert rejected.json()["detail"]["type"] == "validation_error"
    assert await prisma.agentgraphexecution.count(where={"userId": owner}) == 1
    execution_queue.publish_message.assert_awaited_once()

    all_runs = await api.get("/executions")
    graph_runs = await api.get(f"/graphs/{graph_id}/executions")
    assert all_runs.status_code == graph_runs.status_code == 200
    assert [run["id"] for run in all_runs.json()] == [run_id]
    assert [run["id"] for run in graph_runs.json()["executions"]] == [run_id]
    read_back = await get_graph_execution(owner, run_id, include_node_executions=True)
    assert read_back is not None
    field = "value" if source == "raw_input" else "values"
    assert read_back.node_executions[0].input_data[field] == nested(depth)
