import logging

import pytest

from backend.util.test import SpinTestServer

#  NOTE: You can run tests like with the --log-cli-level=INFO to see the logs
# Set up logging
logger = logging.getLogger(__name__)

# Create console handler with formatting
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@pytest.fixture(scope="session")
async def server():
    async with SpinTestServer() as server:
        yield server


@pytest.fixture(scope="session", autouse=True)
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
