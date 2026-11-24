from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from pytest_mock import MockerFixture

from backend.api.features.library import db


@pytest.mark.asyncio
async def test_library_refs_use_the_graph_name_without_a_library_title(
    mocker: MockerFixture,
) -> None:
    names = [
        ("builder", None, "Cat2Dog"),
        ("published", "Marketplace title", "Creator title"),
        ("empty", "", "Other title"),
        ("unnamed", None, None),
    ]
    rows = [
        prisma.models.LibraryAgent.model_construct(
            id=f"library-{graph_id}",
            agentGraphId=graph_id,
            name=library_name,
            AgentGraph=prisma.models.AgentGraph.model_construct(name=graph_name),
            imageUrl=None,
            isDeleted=False,
        )
        for graph_id, library_name, graph_name in names
    ]
    client = MagicMock(find_many=AsyncMock(return_value=rows))
    mocker.patch.object(prisma.models.LibraryAgent, "prisma", return_value=client)

    refs = await db.get_library_agent_refs_by_graph_ids(
        "user-1", [graph_id for graph_id, _, _ in names]
    )

    assert {ref.graph_id: ref.name for ref in refs} == {
        "builder": "Cat2Dog",
        "published": "Marketplace title",
        "empty": "",
        "unnamed": "",
    }
    client.find_many.assert_awaited_once()
    assert client.find_many.call_args.kwargs.get("include") == {"AgentGraph": True}
