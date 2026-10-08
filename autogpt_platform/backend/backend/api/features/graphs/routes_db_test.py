from uuid import uuid4

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.graphs.routes import get_graph, get_graph_all_versions
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_listed_versions_carry_the_credentials_their_sub_graphs_need(
    server: SpinTestServer,
) -> None:
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    sub = await graph_db.create_graph(
        graph_db.Graph(
            name="Sub",
            description="",
            nodes=[graph_db.Node(block_id=ExecuteCodeBlock().id)],
        ),
        user_id,
    )
    parent = await graph_db.create_graph(
        graph_db.Graph(
            name="Parent",
            description="",
            nodes=[
                graph_db.Node(
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
            ],
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

    single = await get_graph(parent.id, user_id, ctx)
    [listed] = await get_graph_all_versions(parent.id, user_id, ctx)

    assert single.credentials_input_schema["properties"]
    assert listed.credentials_input_schema == single.credentials_input_schema
