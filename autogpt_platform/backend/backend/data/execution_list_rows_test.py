"""Reads of many runs at once leave out the inputs each run was started with."""

from uuid import uuid4

import prisma.models
import pytest
from prisma.models import AgentGraph, AgentGraphExecution, LibraryAgent, User

from backend.api.features.library import db as library_db
from backend.api.features.library._get_many import get_library_agents_by_ids
from backend.data.execution import (
    MAX_LISTED_RUN_INPUT_BYTES,
    GraphExecutionListRow,
    get_graph_executions_paginated,
)
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

_INPUT_COLUMNS = {"inputs", "credentialInputs", "nodesInputMasks"}
_CREDENTIALS = {
    "key": {"id": "cred-1", "provider": "github", "type": "api_key", "title": "Mine"}
}


def test_a_list_row_has_every_run_column_but_the_inputs():
    # Relations are capitalised in schema.prisma; columns are not.
    columns = {
        name
        for name in prisma.models.AgentGraphExecution.model_fields
        if not name[0].isupper()
    }
    assert set(GraphExecutionListRow.model_fields) == columns - _INPUT_COLUMNS


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_list_adds_inputs_only_when_asked_and_small(
    server: SpinTestServer,
):
    user_id, graph_id = await _user_with_graph()
    small = await _run(user_id, graph_id, {"topic": "news"}, _CREDENTIALS)
    large = await _run(
        user_id, graph_id, {"text": "x" * (MAX_LISTED_RUN_INPUT_BYTES + 1)}, {}
    )

    listed = await get_graph_executions_paginated(user_id, graph_id=graph_id)
    with_inputs = await get_graph_executions_paginated(
        user_id, graph_id=graph_id, include_inputs=True
    )

    assert {r.id for r in listed.executions} == {small, large}
    for run in listed.executions:
        assert (run.inputs, run.credential_inputs, run.nodes_input_masks) == (
            None,
            None,
            None,
        )
        assert run.graph_id == graph_id
    by_id = {r.id: r for r in with_inputs.executions}
    assert by_id[small].inputs == {"topic": "news"}
    assert by_id[small].credential_inputs is not None
    assert by_id[small].credential_inputs["key"].id == "cred-1"
    assert by_id[large].inputs is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_library_agent_shows_its_latest_runs(server: SpinTestServer):
    user_id, graph_id = await _user_with_graph()
    agent = await LibraryAgent.prisma().create(
        data={
            "userId": user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "name": "Digest",
            "isCreatedByUser": True,
        }
    )
    for _ in range(2):
        await _run(user_id, graph_id, {"topic": "news"}, {})

    single = await library_db.get_library_agent(agent.id, user_id)
    (batched,) = await get_library_agents_by_ids([agent.id], user_id)

    for read in (single, batched):
        assert [r.status for r in read.recent_executions] == ["COMPLETED"] * 2


async def _user_with_graph() -> tuple[str, str]:
    user_id = f"list-rows-{uuid4()}"
    graph_id = str(uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Lister"}
    )
    await AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "Digest",
            "description": "test",
            "userId": user_id,
            "isActive": True,
        }
    )
    return user_id, graph_id


async def _run(user_id: str, graph_id: str, inputs: dict, credentials: dict) -> str:
    run = await AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "userId": user_id,
            "inputs": SafeJson(inputs),
            "credentialInputs": SafeJson(credentials),
        }
    )
    return run.id
