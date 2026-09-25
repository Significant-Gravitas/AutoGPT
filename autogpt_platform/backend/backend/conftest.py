import inspect
import logging
import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from backend.util.logging import configure_logging

load_dotenv()

#  NOTE: You can run tests like with the --log-cli-level=INFO to see the logs
# Set up logging
configure_logging()
logger = logging.getLogger(__name__)

# Reduce Prisma log spam unless PRISMA_DEBUG is set
if not os.getenv("PRISMA_DEBUG"):
    prisma_logger = logging.getLogger("prisma")
    prisma_logger.setLevel(logging.INFO)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    from backend.util.test import SpinTestServer

    async with SpinTestServer() as server:
        yield server


@pytest.fixture(autouse=True)
def no_force_all_flags(monkeypatch: pytest.MonkeyPatch):
    """``load_dotenv()`` above copies ``.env`` into ``os.environ``, so a developer
    who turned on ``FORCE_ALL_FLAGS`` locally would otherwise fail every
    flag-off test. Tests that want the switch on set it themselves."""
    monkeypatch.delenv("FORCE_ALL_FLAGS", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_FORCE_ALL_FLAGS", raising=False)


@pytest.fixture
def test_user_id() -> str:
    """Test user ID fixture."""
    return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture
def admin_user_id() -> str:
    """Admin user ID fixture."""
    return "4e53486c-cf57-477e-ba2a-cb02dc828e1b"


@pytest.fixture
def target_user_id() -> str:
    """Target user ID fixture."""
    return "5e53486c-cf57-477e-ba2a-cb02dc828e1c"


async def _create_user_with_loop_retry(user_data: dict) -> None:
    """Create a user, retrying once on a transient ``Event loop is closed``.

    Fire-and-forget background tasks elsewhere can leave the Prisma pool
    bound to a now-closed test function loop. The first session-loop DB
    call after that surfaces as ``RuntimeError: Event loop is closed``;
    the pool re-establishes itself on the retry.
    """
    from backend.data.user import get_or_create_user
    from backend.util.exceptions import DatabaseError

    try:
        await get_or_create_user(user_data)
    except DatabaseError as e:
        if "Event loop is closed" not in str(e):
            raise
        await get_or_create_user(user_data)


@pytest.fixture
async def setup_test_user(test_user_id):
    """Create test user in database before tests."""
    user_data = {
        "sub": test_user_id,
        "email": "test@example.com",
        "user_metadata": {"name": "Test User"},
    }
    await _create_user_with_loop_retry(user_data)
    return test_user_id


@pytest.fixture
async def setup_admin_user(admin_user_id):
    """Create admin user in database before tests."""
    user_data = {
        "sub": admin_user_id,
        "email": "test-admin@example.com",
        "user_metadata": {"name": "Test Admin"},
    }
    await _create_user_with_loop_retry(user_data)
    return admin_user_id


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup(server):
    """Delete the graphs and store listings that tests created through the test
    server, at the end of the session, so they don't pile up in the test DB."""
    created_graphs: list[tuple[str, str]] = []
    created_listing_ids: list[str] = []
    agent_server = server.agent_server
    original_create_graph = agent_server.test_create_graph
    original_create_store_listing = agent_server.test_create_store_listing

    async def create_graph_wrapper(*args, **kwargs):
        created_graph = await original_create_graph(*args, **kwargs)
        # Callers pass user_id both positionally and by keyword.
        call = inspect.signature(original_create_graph).bind(*args, **kwargs)
        user_id = call.arguments["user_id"]
        created_graphs.extend(
            (graph.id, user_id) for graph in [created_graph, *created_graph.sub_graphs]
        )
        return created_graph

    async def create_store_listing_wrapper(*args, **kwargs):
        from fastapi.responses import JSONResponse

        store_listing = await original_create_store_listing(*args, **kwargs)
        if not isinstance(store_listing, JSONResponse):
            created_listing_ids.append(store_listing.listing_id)
        return store_listing

    try:
        agent_server.test_create_graph = create_graph_wrapper
        agent_server.test_create_store_listing = create_store_listing_wrapper
        yield  # This runs the test function
    finally:
        agent_server.test_create_graph = original_create_graph
        agent_server.test_create_store_listing = original_create_store_listing
        await _delete_test_graphs(agent_server, created_graphs, created_listing_ids)


async def _delete_test_graphs(
    agent_server, created_graphs: list[tuple[str, str]], listing_ids: list[str]
) -> None:
    from prisma.models import (
        AgentGraph,
        AgentNodeExecutionInputOutput,
        AgentPreset,
        LibraryAgent,
        StoreListing,
    )

    graph_ids = [graph_id for graph_id, _ in created_graphs]
    # Listing versions, presets and every user's library entries hold the graph
    # (onDelete: Restrict), so they go first. Deleting a listing cascades to
    # its versions and their reviews; a preset's saved inputs would only be
    # orphaned, so they are deleted with it.
    await StoreListing.prisma().delete_many(where={"id": {"in": listing_ids}})
    await LibraryAgent.prisma().delete_many(where={"agentGraphId": {"in": graph_ids}})
    presets = await AgentPreset.prisma().find_many(
        where={"agentGraphId": {"in": graph_ids}}
    )
    preset_ids = [preset.id for preset in presets]
    await AgentNodeExecutionInputOutput.prisma().delete_many(
        where={"agentPresetId": {"in": preset_ids}}
    )
    await AgentPreset.prisma().delete_many(where={"id": {"in": preset_ids}})
    for graph_id, user_id in created_graphs:
        await agent_server.test_delete_graph(graph_id, user_id)

    remaining = await AgentGraph.prisma().find_many(where={"id": {"in": graph_ids}})
    leftover = sorted({graph.id for graph in remaining})
    assert not leftover, f"Test graphs were not deleted: {leftover}"
