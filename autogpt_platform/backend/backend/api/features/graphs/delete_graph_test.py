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
import backend.data.graph as graph_db
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


async def _create_preset(created, user_id: str):
    return await prisma.models.AgentPreset.prisma().create(
        data={
            "name": "Delete test preset",
            "description": "",
            "userId": user_id,
            "agentGraphId": created.id,
            "agentGraphVersion": created.version,
        }
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
    # The owner's presets go with it, and so do their saved inputs, which
    # the preset FK would only orphan (ON DELETE SET NULL).
    preset = await _create_preset(created, DEFAULT_USER_ID)
    preset_input = await prisma.models.AgentNodeExecutionInputOutput.prisma().create(
        data={"name": "input", "agentPresetId": preset.id}
    )

    result = await server.agent_server.test_delete_graph(created.id, DEFAULT_USER_ID)

    assert result == {"version_counts": 1}
    assert not await _graph_exists(created.id)
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 0
    assert not await prisma.models.AgentGraphExecution.prisma().find_unique(
        where={"id": run.id}
    )
    assert not await prisma.models.AgentPreset.prisma().find_unique(
        where={"id": preset.id}
    )
    assert not await prisma.models.AgentNodeExecutionInputOutput.prisma().find_unique(
        where={"id": preset_input.id}
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_graph_keeps_versions_saved_under_another_org(
    server: SpinTestServer,
):
    created = await _create_graph(server)
    # The owner saved a later version while another org was active.
    other_org_version = await prisma.models.AgentGraph.prisma().create(
        data={
            "id": created.id,
            "version": created.version + 1,
            "name": created.name,
            "isActive": False,
            "userId": DEFAULT_USER_ID,
            "organizationId": f"other-org-{uuid4()}",
        }
    )
    other_org_preset = await prisma.models.AgentPreset.prisma().create(
        data={
            "name": "Other org preset",
            "description": "",
            "userId": DEFAULT_USER_ID,
            "agentGraphId": created.id,
            "agentGraphVersion": other_org_version.version,
        }
    )

    try:
        result = await server.agent_server.test_delete_graph(
            created.id, DEFAULT_USER_ID
        )

        assert result == {"version_counts": 1}
        assert (
            await prisma.models.AgentGraph.prisma().count(where={"id": created.id}) == 1
        )
        assert await prisma.models.AgentPreset.prisma().find_unique(
            where={"id": other_org_preset.id}
        )
    finally:
        await prisma.models.AgentPreset.prisma().delete_many(
            where={"id": other_org_preset.id}
        )
        await prisma.models.AgentGraph.prisma().delete_many(
            where={"id": created.id, "version": other_org_version.version}
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


async def _add_other_users_library_entry(created, user_id: str) -> None:
    await prisma.models.LibraryAgent.prisma().create(
        data={
            "userId": user_id,
            "agentGraphId": created.id,
            "agentGraphVersion": created.version,
        }
    )


async def _add_other_users_run(created, user_id: str) -> None:
    await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": created.id,
            "agentGraphVersion": created.version,
            "userId": user_id,
        }
    )


async def _other_users_references(graph_id: str, user_id: str) -> int:
    where: dict = {"agentGraphId": graph_id, "userId": user_id}
    return (
        await prisma.models.LibraryAgent.prisma().count(where=where)
        + await prisma.models.AgentPreset.prisma().count(where=where)
        + await prisma.models.AgentGraphExecution.prisma().count(where=where)
    )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "add_reference",
    [_add_other_users_library_entry, _create_preset, _add_other_users_run],
    ids=["library_entry", "preset", "run"],
)
async def test_delete_graph_used_by_another_user_is_refused(
    server: SpinTestServer, add_reference
):
    created = await _create_graph(server)
    other_user = await create_test_user(alt_user=True)
    await add_reference(created, other_user.id)

    await _assert_refused(server, created.id)
    assert await _other_users_references(created.id, other_user.id) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_graph_referenced_after_the_check_is_refused(
    server: SpinTestServer,
):
    created = await _create_graph(server)
    other_user = await create_test_user(alt_user=True)
    await _add_other_users_library_entry(created, other_user.id)

    # Both checks miss the reference, as when it lands between them and the
    # delete: the FK backstop still turns it into a 409.
    with (
        patch.object(graph_db, "assert_graph_deletable", return_value=None),
        pytest.raises(fastapi.HTTPException) as exc_info,
    ):
        await server.agent_server.test_delete_graph(created.id, DEFAULT_USER_ID)

    assert exc_info.value.status_code == 409
    assert await _graph_exists(created.id)
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 1


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


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_graph_by_non_owner_changes_nothing(server: SpinTestServer):
    created = await _create_graph(server)
    other_user = await create_test_user(alt_user=True)

    with patch.object(
        graphs_routes,
        "on_graph_deactivate",
        wraps=graphs_routes.on_graph_deactivate,
    ) as deactivate:
        result = await server.agent_server.test_delete_graph(created.id, other_user.id)

    assert result == {"version_counts": 0}
    deactivate.assert_not_called()
    assert await _graph_exists(created.id)
    assert await _library_rows(created.id, DEFAULT_USER_ID) == 1
