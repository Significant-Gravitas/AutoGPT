"""DB tests for DELETE /graphs/{graph_id} (#15637).

These go through the real route and database: POST /graphs always creates a
library entry for the owner, and that entry (like presets and store listing
versions) holds the graph with an ON DELETE RESTRICT foreign key.
"""

from unittest.mock import patch
from uuid import uuid4

import fastapi
import prisma.models
import prisma.types
import pytest

import backend.api.features.library.db as library_db
import backend.api.features.store.model as store
from backend.api.features.graphs import routes as graphs_routes
from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.data.graph import Graph, Node
from backend.data.user import DEFAULT_USER_ID
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer


async def _create_graph(server: SpinTestServer):
    graph = Graph(
        name=f"Delete me {uuid4()}",
        description="Graph for delete tests",
        nodes=[Node(id="node_1", block_id=StoreValueBlock().id)],
        links=[],
    )
    return await server.agent_server.test_create_graph(
        CreateGraph(graph=graph), DEFAULT_USER_ID
    )


async def _graph_exists(graph_id: str) -> bool:
    return bool(await prisma.models.AgentGraph.prisma().count(where={"id": graph_id}))


async def _library_rows(graph_id: str, user_id: str) -> int:
    return await prisma.models.LibraryAgent.prisma().count(
        where={"agentGraphId": graph_id, "userId": user_id}
    )


async def _assert_refused(server: SpinTestServer, graph_id: str) -> None:
    with (
        patch.object(
            graphs_routes,
            "on_graph_deactivate",
            wraps=graphs_routes.on_graph_deactivate,
        ) as deactivate,
        patch.object(
            library_db,
            "cleanup_graph_for_permanent_delete",
            wraps=library_db.cleanup_graph_for_permanent_delete,
        ) as cleanup,
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            await server.agent_server.test_delete_graph(graph_id, DEFAULT_USER_ID)

    assert exc_info.value.status_code == 409
    # A refused delete leaves the graph and everything that drives it alone.
    deactivate.assert_not_called()
    cleanup.assert_not_called()
    assert await _graph_exists(graph_id)
    assert await _library_rows(graph_id, DEFAULT_USER_ID) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_created_graph(server: SpinTestServer):
    created = await _create_graph(server)
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 1
    # The owner's run history goes with the graph (AgentGraphExecution
    # cascades on AgentGraph).
    run = await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": created.id,
            "agentGraphVersion": created.version,
            "userId": DEFAULT_USER_ID,
        }
    )

    result = await server.agent_server.test_delete_graph(created.id, DEFAULT_USER_ID)

    assert result == {"version_counts": 1}
    assert not await _graph_exists(created.id)
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 0
    assert not await prisma.models.AgentGraphExecution.prisma().find_unique(
        where={"id": run.id}
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_graph_after_library_soft_delete(server: SpinTestServer):
    created = await _create_graph(server)
    library_agent = await prisma.models.LibraryAgent.prisma().find_first_or_raise(
        where={"agentGraphId": created.id, "userId": DEFAULT_USER_ID}
    )
    await library_db.delete_library_agent(
        library_agent.id, DEFAULT_USER_ID, soft_delete=True
    )
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 1  # soft-deleted

    result = await server.agent_server.test_delete_graph(created.id, DEFAULT_USER_ID)

    assert result == {"version_counts": 1}
    assert not await _graph_exists(created.id)
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_graph_in_other_users_library_is_refused(
    server: SpinTestServer,
):
    created = await _create_graph(server)
    other_user = await create_test_user(alt_user=True)
    await prisma.models.LibraryAgent.prisma().create(
        data={
            "userId": other_user.id,
            "agentGraphId": created.id,
            "agentGraphVersion": created.version,
        }
    )

    await _assert_refused(server, created.id)
    assert await _library_rows(created.id, other_user.id) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_graph_with_store_listing_is_refused(server: SpinTestServer):
    created = await _create_graph(server)
    if not await prisma.models.Profile.prisma().find_first(
        where={"userId": DEFAULT_USER_ID}
    ):
        await prisma.models.Profile.prisma().create(
            data=prisma.types.ProfileCreateInput(
                userId=DEFAULT_USER_ID,
                name="Default User",
                username=f"default-user-{DEFAULT_USER_ID[:8]}",
                description="Default test user profile",
                links=[],
            )
        )
    listing = await server.agent_server.test_create_store_listing(
        store.StoreSubmissionRequest(
            graph_id=created.id,
            graph_version=created.version,
            slug=created.id,
            name="Delete test listing",
            sub_heading="Sub heading",
            video_url=None,
            image_urls=[],
            description="Description",
            categories=["operations"],
        ),
        DEFAULT_USER_ID,
    )
    assert not isinstance(listing, fastapi.responses.JSONResponse)

    await _assert_refused(server, created.id)
