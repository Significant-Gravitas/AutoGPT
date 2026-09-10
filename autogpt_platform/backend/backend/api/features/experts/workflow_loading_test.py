from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert


@pytest.mark.asyncio
@pytest.mark.parametrize("with_metrics", [True, False])
async def test_roster_skips_nodes_but_detail_loads_workflow_chain(with_metrics):
    graph = prisma.models.AgentGraph.model_construct(
        name="My workflow", description="Created in the library", Nodes=None
    )
    workflow = prisma.models.ExpertWorkflow.model_construct(
        id="workflow-1",
        storeListingVersionId=None,
        libraryAgentId="library-1",
        LibraryAgent=prisma.models.LibraryAgent.model_construct(
            agentGraphId="graph-1", name=None, description=None, AgentGraph=graph
        ),
        StoreListingVersion=None,
    )
    row = prisma.models.Expert.model_construct(id="expert-1", Workflows=[workflow])
    client = SimpleNamespace(
        find_many=AsyncMock(return_value=[row]), find_first=AsyncMock(return_value=row)
    )

    def convert(expert, *_args):
        return Expert.model_construct(
            id=expert.id,
            workflows=[experts_db._to_workflow_ref(w) for w in expert.Workflows],
        )

    with (
        patch.object(prisma.models.Expert, "prisma", return_value=client),
        patch.object(experts_db, "_latest_runs", new=AsyncMock(return_value={})),
        patch.object(experts_db, "_weekly_spends", new=AsyncMock(return_value={})),
        patch.object(experts_db, "get_weekly_spend", new=AsyncMock(return_value=0)),
        patch.object(
            experts_db, "count_expert_credentials", new=AsyncMock(return_value={})
        ),
        patch.object(experts_db, "_to_model", side_effect=convert),
    ):
        roster = await experts_db.list_experts("owner-1", with_metrics=with_metrics)
        graph.Nodes = [
            prisma.models.AgentNode.model_construct(
                agentBlockId="c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
                constantInput="{}",
            )
        ]
        detail = await experts_db.get_expert("owner-1", row.id)

    roster_graph = client.find_many.await_args.kwargs["include"]["Workflows"][
        "include"
    ]["LibraryAgent"]["include"]["AgentGraph"]
    detail_graph = client.find_first.await_args.kwargs["include"]["Workflows"][
        "include"
    ]["LibraryAgent"]["include"]["AgentGraph"]
    assert roster_graph is True
    assert detail_graph["include"]["Nodes"] is True
    assert roster[0].workflows[0].name == "My workflow"
    assert roster[0].workflows[0].description == "Created in the library"
    assert roster[0].workflows[0].chain == []
    assert detail is not None
    assert [item.kind for item in detail.workflows[0].chain] == ["input"]
