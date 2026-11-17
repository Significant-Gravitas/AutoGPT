import asyncio
import uuid

import pytest

from backend.blocks.jina.search import SearchTheWebBlock
from backend.data.graph import Graph, Node
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import GraphActivationError
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_credential_reaches_the_rpc_caller_at_once(
    server: SpinTestServer, setup_test_user: str
):
    """The copilot saves agents through the DatabaseManager; a refused
    activation must arrive as itself, not as a 500 the client retries."""
    graph = Graph(
        name="Search with a deleted key",
        description="",
        nodes=[
            Node(
                block_id=SearchTheWebBlock().id,
                input_default={
                    "credentials": {
                        "id": str(uuid.uuid4()),
                        "provider": "jina",
                        "type": "api_key",
                        "title": "Deleted Jina key",
                    },
                    "query": "anything",
                },
            )
        ],
    )
    client = get_database_manager_async_client()
    # Untimed: the server may still be starting, and that is not what is measured.
    await client.health_check_async()

    with pytest.raises(GraphActivationError, match="no longer exists"):
        await asyncio.wait_for(
            client.create_graph_in_library(graph, setup_test_user), timeout=30
        )
