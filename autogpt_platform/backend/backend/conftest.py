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


@pytest.fixture
async def setup_test_user(test_user_id):
    """Create test user in database before tests."""
    from backend.data.user import get_or_create_user

    # Create the test user in the database using JWT token format
    user_data = {
        "sub": test_user_id,
        "email": "test@example.com",
        "user_metadata": {"name": "Test User"},
    }
    await get_or_create_user(user_data)
    return test_user_id


@pytest.fixture
async def setup_admin_user(admin_user_id):
    """Create admin user in database before tests."""
    from backend.data.user import get_or_create_user

    # Create the admin user in the database using JWT token format
    user_data = {
        "sub": admin_user_id,
        "email": "test-admin@example.com",
        "user_metadata": {"name": "Test Admin"},
    }
    await get_or_create_user(user_data)
    return admin_user_id


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup(server):
    created_graph_ids = []
    original_create_graph = server.agent_server.test_create_graph

    async def create_graph_wrapper(*args, **kwargs):
        created_graph = await original_create_graph(*args, **kwargs)
        # Extract user_id correctly
        user_id = kwargs.get("user_id", args[2] if len(args) > 2 else None)
        created_graph_ids.append((created_graph.id, user_id))
        return created_graph

    try:
        server.agent_server.test_create_graph = create_graph_wrapper
        yield  # This runs the test function
    finally:
        server.agent_server.test_create_graph = original_create_graph

        # Delete the created graphs and assert they were deleted
        for graph_id, user_id in created_graph_ids:
            if user_id:
                resp = await server.agent_server.test_delete_graph(graph_id, user_id)
                num_deleted = resp["version_counts"]
                assert num_deleted > 0, f"Graph {graph_id} was not deleted."
