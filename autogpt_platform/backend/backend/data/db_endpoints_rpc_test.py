import asyncio
import uuid

import pytest

from backend.data import graph as graph_db
from backend.data.db_endpoints import declared_endpoints
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import GraphNotAccessibleError
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_an_endpoint_a_module_declares_answers_and_refuses_over_the_rpc(
    server: SpinTestServer, setup_test_user: str
):
    """The manager serves these from the module's declaration alone: one must
    answer through the real client, and its own error must reach the caller."""
    assert {"get_graph_metadata", "validate_graph_execution_permissions"} <= set(
        declared_endpoints(graph_db)
    )
    client = get_database_manager_async_client()
    # Untimed: the server may still be starting, and that is not what is measured.
    await client.health_check_async()
    missing_graph = str(uuid.uuid4())

    assert (
        await asyncio.wait_for(client.get_graph_metadata(missing_graph), timeout=20)
        is None
    )
    with pytest.raises(GraphNotAccessibleError):
        await asyncio.wait_for(
            client.validate_graph_execution_permissions(
                user_id=setup_test_user, graph_id=missing_graph, graph_version=1
            ),
            timeout=20,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_the_less_plain_endpoints_answer_over_the_rpc(
    server: SpinTestServer, setup_test_user: str
):
    """An adapter the manager module declares for itself, a function behind a
    caching decorator, and one renamed to its route: each is served."""
    client = get_database_manager_async_client()
    await client.health_check_async()

    balance = await asyncio.wait_for(client.get_credits(setup_test_user), timeout=20)
    user = await asyncio.wait_for(client.get_user_by_id(setup_test_user), timeout=20)
    await asyncio.wait_for(
        client.increment_push_fail_count(setup_test_user, "https://push.invalid/none"),
        timeout=20,
    )

    assert isinstance(balance, int)
    assert user.id == setup_test_user
