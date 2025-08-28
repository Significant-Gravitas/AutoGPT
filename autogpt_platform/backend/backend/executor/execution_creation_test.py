"""Test execution creation with proper ID generation and persistence."""

import asyncio
import threading
import uuid
from datetime import datetime

import pytest

from backend.data.execution import ExecutionStatus
from backend.executor.execution_data import ExecutionDataClient


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def execution_client_with_mock_db(event_loop):
    """Create an ExecutionDataClient with mocked database operations."""
    # Mock the database methods to avoid external service dependencies
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock, patch

    from backend.data.execution import ExecutionStatus, GraphExecutionMeta

    # Mock the graph execution metadata - align with assertions below
    mock_graph_meta = GraphExecutionMeta(
        id="test_graph_exec_id",
        user_id="test_user_123",
        graph_id="test_graph_456",
        graph_version=1,
        status=ExecutionStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        stats=None,
    )

    # Create client with ThreadPoolExecutor and graph metadata (constructed inside patch)
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)

    # Storage for tracking created executions
    created_executions = []

    async def mock_create_node_execution(
        node_exec_id, node_id, graph_exec_id, input_name, input_data
    ):
        """Mock execution creation that records what was created."""
        created_executions.append(
            {
                "node_exec_id": node_exec_id,
                "node_id": node_id,
                "graph_exec_id": graph_exec_id,
                "input_name": input_name,
                "input_data": input_data,
            }
        )
        return node_exec_id

    def sync_mock_create_node_execution(
        node_exec_id, node_id, graph_exec_id, input_name, input_data
    ):
        """Mock sync execution creation that records what was created."""
        created_executions.append(
            {
                "node_exec_id": node_exec_id,
                "node_id": node_id,
                "graph_exec_id": graph_exec_id,
                "input_name": input_name,
                "input_data": input_data,
            }
        )
        return node_exec_id

    # Prepare mock async and sync DB clients
    async_mock_client = AsyncMock()
    async_mock_client.create_node_execution = mock_create_node_execution

    sync_mock_client = MagicMock()
    sync_mock_client.create_node_execution = sync_mock_create_node_execution
    # Mock graph execution for return values
    from backend.data.execution import GraphExecutionMeta

    mock_graph_update = GraphExecutionMeta(
        id="test_graph_exec_id",
        user_id="test_user_123",
        graph_id="test_graph_456",
        graph_version=1,
        status=ExecutionStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        stats=None,
    )

    # No-ops for other sync methods used by the client during tests
    sync_mock_client.add_input_to_node_execution.side_effect = lambda **kwargs: None
    sync_mock_client.update_node_execution_status.side_effect = (
        lambda *args, **kwargs: None
    )
    sync_mock_client.upsert_execution_output.side_effect = lambda **kwargs: None
    sync_mock_client.update_graph_execution_stats.side_effect = (
        lambda *args, **kwargs: mock_graph_update
    )
    sync_mock_client.update_graph_execution_start_time.side_effect = (
        lambda *args, **kwargs: mock_graph_update
    )

    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()
    with patch(
        "backend.executor.execution_data.get_database_manager_async_client",
        return_value=async_mock_client,
    ), patch(
        "backend.executor.execution_data.get_database_manager_client",
        return_value=sync_mock_client,
    ):
        # Now construct the client under the patch so it captures the mocked clients
        client = ExecutionDataClient(executor, "test_graph_exec_id", mock_graph_meta)
        # Store the mocks for the test to access if needed
        setattr(client, "_test_async_client", async_mock_client)
        setattr(client, "_test_sync_client", sync_mock_client)
        setattr(client, "_created_executions", created_executions)
        yield client

    # Cleanup
    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(timeout=1)


class TestExecutionCreation:
    """Test execution creation with proper ID generation and persistence."""

    async def test_execution_creation_with_valid_ids(
        self, execution_client_with_mock_db
    ):
        """Test that execution creation generates and persists valid IDs."""
        client = execution_client_with_mock_db

        node_id = "test_node_789"
        input_name = "test_input"
        input_data = "test_value"
        block_id = "test_block_abc"

        # This should trigger execution creation since cache is empty
        exec_id, input_dict = client.upsert_execution_input(
            node_id=node_id,
            input_name=input_name,
            input_data=input_data,
            block_id=block_id,
        )

        # Verify execution ID is valid UUID
        try:
            uuid.UUID(exec_id)
        except ValueError:
            pytest.fail(f"Generated execution ID '{exec_id}' is not a valid UUID")

        # Verify execution was created in cache with complete data
        assert exec_id in client._cache._node_executions
        cached_execution = client._cache._node_executions[exec_id]

        # Check all required fields have valid values
        assert cached_execution.user_id == "test_user_123"
        assert cached_execution.graph_id == "test_graph_456"
        assert cached_execution.graph_version == 1
        assert cached_execution.graph_exec_id == "test_graph_exec_id"
        assert cached_execution.node_exec_id == exec_id
        assert cached_execution.node_id == node_id
        assert cached_execution.block_id == block_id
        assert cached_execution.status == ExecutionStatus.INCOMPLETE
        assert cached_execution.input_data == {input_name: input_data}
        assert isinstance(cached_execution.add_time, datetime)

        # Verify execution was persisted to database with our generated ID
        created_executions = getattr(client, "_created_executions", [])
        assert len(created_executions) == 1
        created = created_executions[0]
        assert created["node_exec_id"] == exec_id  # Our generated ID was used
        assert created["node_id"] == node_id
        assert created["graph_exec_id"] == "test_graph_exec_id"
        assert created["input_name"] == input_name
        assert created["input_data"] == input_data

        # Verify input dict returned correctly
        assert input_dict == {input_name: input_data}

    async def test_execution_reuse_vs_creation(self, execution_client_with_mock_db):
        """Test that execution reuse works and creation only happens when needed."""
        client = execution_client_with_mock_db

        node_id = "reuse_test_node"
        block_id = "reuse_test_block"

        # Create first execution
        exec_id_1, input_dict_1 = client.upsert_execution_input(
            node_id=node_id,
            input_name="input_1",
            input_data="value_1",
            block_id=block_id,
        )

        # This should reuse the existing INCOMPLETE execution
        exec_id_2, input_dict_2 = client.upsert_execution_input(
            node_id=node_id,
            input_name="input_2",
            input_data="value_2",
            block_id=block_id,
        )

        # Should reuse the same execution
        assert exec_id_1 == exec_id_2
        assert input_dict_2 == {"input_1": "value_1", "input_2": "value_2"}

        # Only one execution should be created in database
        created_executions = getattr(client, "_created_executions", [])
        assert len(created_executions) == 1

        # Verify cache has the merged inputs
        cached_execution = client._cache._node_executions[exec_id_1]
        assert cached_execution.input_data == {
            "input_1": "value_1",
            "input_2": "value_2",
        }

        # Now complete the execution and try to add another input
        client.update_node_status_and_publish(
            exec_id=exec_id_1, status=ExecutionStatus.COMPLETED
        )

        # Verify the execution status was actually updated in the cache
        updated_execution = client._cache._node_executions[exec_id_1]
        assert (
            updated_execution.status == ExecutionStatus.COMPLETED
        ), f"Expected COMPLETED but got {updated_execution.status}"

        # This should create a NEW execution since the first is no longer INCOMPLETE
        exec_id_3, input_dict_3 = client.upsert_execution_input(
            node_id=node_id,
            input_name="input_3",
            input_data="value_3",
            block_id=block_id,
        )

        # Should be a different execution
        assert exec_id_3 != exec_id_1
        assert input_dict_3 == {"input_3": "value_3"}

        # Verify cache behavior: should have two different executions in cache now
        cached_executions = client._cache._node_executions
        assert len(cached_executions) == 2
        assert exec_id_1 in cached_executions
        assert exec_id_3 in cached_executions

        # First execution should be COMPLETED
        assert cached_executions[exec_id_1].status == ExecutionStatus.COMPLETED
        # Third execution should be INCOMPLETE (newly created)
        assert cached_executions[exec_id_3].status == ExecutionStatus.INCOMPLETE

    async def test_multiple_nodes_get_different_execution_ids(
        self, execution_client_with_mock_db
    ):
        """Test that different nodes get different execution IDs."""
        client = execution_client_with_mock_db

        # Create executions for different nodes
        exec_id_a, _ = client.upsert_execution_input(
            node_id="node_a",
            input_name="test_input",
            input_data="test_value",
            block_id="block_a",
        )

        exec_id_b, _ = client.upsert_execution_input(
            node_id="node_b",
            input_name="test_input",
            input_data="test_value",
            block_id="block_b",
        )

        # Should be different executions with different IDs
        assert exec_id_a != exec_id_b

        # Both should be valid UUIDs
        uuid.UUID(exec_id_a)
        uuid.UUID(exec_id_b)

        # Both should be in cache
        cached_executions = client._cache._node_executions
        assert len(cached_executions) == 2
        assert exec_id_a in cached_executions
        assert exec_id_b in cached_executions

        # Both should have correct node IDs
        assert cached_executions[exec_id_a].node_id == "node_a"
        assert cached_executions[exec_id_b].node_id == "node_b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
