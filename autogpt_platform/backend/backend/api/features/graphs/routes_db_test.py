from uuid import uuid4

import pytest
from autogpt_libs.auth.models import RequestContext
from prisma.actions import AgentGraphActions
from pytest_mock import MockerFixture

from backend.api.features.graphs.routes import get_graph, get_graph_all_versions
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.blocks.github.issues import GithubCommentBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_listed_versions_carry_the_credentials_their_sub_graphs_need(
    server: SpinTestServer, mocker: MockerFixture
) -> None:
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    code, github = [
        await graph_db.create_graph(
            graph_db.Graph(
                name=block.name,
                description="",
                nodes=[graph_db.Node(block_id=block.id)],
            ),
            user_id,
        )
        for block in (ExecuteCodeBlock(), GithubCommentBlock())
    ]
    parent_id = str(uuid4())
    # v1 runs one sub-graph and v2 both, so a version given the other's sub-graphs shows
    for version, subs in ((1, [code]), (2, [code, github])):
        await graph_db.create_graph(
            graph_db.Graph(
                id=parent_id,
                version=version,
                name="Parent",
                description="",
                nodes=[_agent_executor_node(user_id, sub) for sub in subs],
            ),
            user_id,
        )
    ctx = RequestContext(
        user_id=user_id,
        org_id=str(uuid4()),
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )

    find_many = mocker.spy(AgentGraphActions, "find_many")
    listed = {
        v.version: v for v in await get_graph_all_versions(parent_id, user_id, ctx)
    }
    sub_graph_fetches = [
        c for c in find_many.call_args_list if "OR" in c.kwargs["where"]
    ]

    assert len(sub_graph_fetches) == 1
    assert {v: {s.id for s in g.sub_graphs} for v, g in listed.items()} == {
        1: {code.id},
        2: {code.id, github.id},
    }
    assert listed[1].credentials_input_schema != listed[2].credentials_input_schema
    for version, graph in listed.items():
        single = await get_graph(parent_id, user_id, ctx, version=version)
        assert graph.credentials_input_schema == single.credentials_input_schema


def _agent_executor_node(user_id: str, sub: graph_db.GraphModel) -> graph_db.Node:
    return graph_db.Node(
        block_id=AgentExecutorBlock().id,
        input_default={
            "user_id": user_id,
            "graph_id": sub.id,
            "graph_version": sub.version,
            "inputs": {},
            "input_schema": {},
            "output_schema": {},
        },
    )
