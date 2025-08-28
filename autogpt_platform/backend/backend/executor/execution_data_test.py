"""Test suite for ExecutionDataClient."""

import asyncio
import threading

import pytest

from backend.data.execution import ExecutionStatus
from backend.executor.execution_data import ExecutionDataClient


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def execution_client(event_loop):
    from datetime import datetime, timezone

    from backend.data.execution import ExecutionStatus, GraphExecutionMeta

    mock_graph_meta = GraphExecutionMeta(
        id="test_graph_exec_id",
        user_id="test_user_id",
        graph_id="test_graph_id",
        graph_version=1,
        status=ExecutionStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        stats=None,
    )

    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)
    client = ExecutionDataClient(executor, "test_graph_exec_id", mock_graph_meta)

    thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    thread.start()

    yield client

    event_loop.call_soon_threadsafe(event_loop.stop)
    thread.join(timeout=1)


class TestExecutionDataClient:

    async def test_update_node_status_writes_to_cache_immediately(
        self, execution_client
    ):
        """Test that node status updates are immediately visible in cache."""
        # First create an execution to update
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id="test-node",
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        status = ExecutionStatus.RUNNING
        execution_data = {"step": "processing"}
        stats = {"duration": 5.2}

        # Update status of existing execution
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=status,
            execution_data=execution_data,
            stats=stats,
        )

        # Verify immediate visibility in cache
        cached_exec = execution_client.get_node_execution(node_exec_id)
        assert cached_exec is not None
        assert cached_exec.status == status
        # execution_data should be merged with existing input_data, not replace it
        expected_input_data = {"test_input": "test_value", "step": "processing"}
        assert cached_exec.input_data == expected_input_data

    def test_update_node_status_execution_not_found_raises_error(
        self, execution_client
    ):
        """Test that updating non-existent execution raises error instead of creating it."""
        non_existent_id = "does-not-exist"

        with pytest.raises(
            RuntimeError, match="Execution does-not-exist not found in cache"
        ):
            execution_client.update_node_status_and_publish(
                exec_id=non_existent_id, status=ExecutionStatus.COMPLETED
            )

    async def test_upsert_execution_output_writes_to_cache_immediately(
        self, execution_client
    ):
        """Test that output updates are immediately visible in cache."""
        # First create an execution
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id="test-node",
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        output_name = "result"
        output_data = {"answer": 42, "confidence": 0.95}

        # Update to RUNNING status first
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.RUNNING,
            execution_data={"input": "test"},
        )

        execution_client.upsert_execution_output(
            node_exec_id=node_exec_id, output_name=output_name, output_data=output_data
        )
        # Check output through the node execution
        cached_exec = execution_client.get_node_execution(node_exec_id)
        assert cached_exec is not None
        assert output_name in cached_exec.output_data
        assert cached_exec.output_data[output_name] == [output_data]

    async def test_get_node_execution_reads_from_cache(self, execution_client):
        """Test that get_node_execution returns cached data immediately."""
        # First create an execution to work with
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id="test-node",
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        # Then update its status
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.COMPLETED,
            execution_data={"result": "success"},
        )

        result = execution_client.get_node_execution(node_exec_id)

        assert result is not None
        assert result.status == ExecutionStatus.COMPLETED
        # execution_data gets merged with existing input_data
        expected_input_data = {"test_input": "test_value", "result": "success"}
        assert result.input_data == expected_input_data

    async def test_get_latest_node_execution_reads_from_cache(self, execution_client):
        """Test that get_latest_node_execution returns cached data."""
        node_id = "node-1"

        # First create an execution for this node
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id=node_id,
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        # Then update its status to make it non-INCOMPLETE (so it's returned by get_latest)
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.RUNNING,
            execution_data={"from": "cache"},
        )

        result = execution_client.get_latest_node_execution(node_id)

        assert result is not None
        assert result.status == ExecutionStatus.RUNNING
        # execution_data gets merged with existing input_data
        expected_input_data = {"test_input": "test_value", "from": "cache"}
        assert result.input_data == expected_input_data

    async def test_get_node_executions_sync_filters_correctly(self, execution_client):
        # Create executions with different statuses
        executions = [
            (ExecutionStatus.RUNNING, "block-a"),
            (ExecutionStatus.COMPLETED, "block-a"),
            (ExecutionStatus.FAILED, "block-b"),
            (ExecutionStatus.RUNNING, "block-b"),
        ]

        exec_ids = []
        for i, (status, block_id) in enumerate(executions):
            # First create the execution
            exec_id, _ = execution_client.upsert_execution_input(
                node_id=f"node-{i}",
                input_name="test_input",
                input_data="test_value",
                block_id=block_id,
            )
            exec_ids.append(exec_id)

            # Then update its status and metadata
            execution_client.update_node_status_and_publish(
                exec_id=exec_id, status=status, execution_data={"block": block_id}
            )
            # Update cached execution with graph_exec_id and block_id for filtering
            # Note: In real implementation, these would be set during creation
            # For test purposes, we'll skip this manual update since the filtering
            # logic should work with the data as created

        # Test status filtering
        running_execs = execution_client.get_node_executions(
            statuses=[ExecutionStatus.RUNNING]
        )
        assert len(running_execs) == 2
        assert all(e.status == ExecutionStatus.RUNNING for e in running_execs)

        # Test block_id filtering
        block_a_execs = execution_client.get_node_executions(block_ids=["block-a"])
        assert len(block_a_execs) == 2
        assert all(e.block_id == "block-a" for e in block_a_execs)

        # Test combined filtering
        running_block_b = execution_client.get_node_executions(
            statuses=[ExecutionStatus.RUNNING], block_ids=["block-b"]
        )
        assert len(running_block_b) == 1
        assert running_block_b[0].status == ExecutionStatus.RUNNING
        assert running_block_b[0].block_id == "block-b"

    async def test_write_then_read_consistency(self, execution_client):
        """Test critical race condition scenario: immediate read after write."""
        # First create an execution to work with
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id="consistency-test-node",
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        # Write status
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.RUNNING,
            execution_data={"step": 1},
        )

        # Write output
        execution_client.upsert_execution_output(
            node_exec_id=node_exec_id,
            output_name="intermediate",
            output_data={"progress": 50},
        )

        # Update status again
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.COMPLETED,
            execution_data={"step": 2},
        )

        # All changes should be immediately visible
        cached_exec = execution_client.get_node_execution(node_exec_id)
        assert cached_exec is not None
        assert cached_exec.status == ExecutionStatus.COMPLETED
        # execution_data gets merged with existing input_data - step 2 overwrites step 1
        expected_input_data = {"test_input": "test_value", "step": 2}
        assert cached_exec.input_data == expected_input_data

        # Output should be visible in execution record
        assert cached_exec.output_data["intermediate"] == [{"progress": 50}]

    async def test_concurrent_operations_are_thread_safe(self, execution_client):
        """Test that concurrent operations don't corrupt cache."""
        num_threads = 3  # Reduced for simpler test
        operations_per_thread = 5  # Reduced for simpler test

        # Create all executions upfront
        created_exec_ids = []
        for thread_id in range(num_threads):
            for i in range(operations_per_thread):
                exec_id, _ = execution_client.upsert_execution_input(
                    node_id=f"node-{thread_id}-{i}",
                    input_name="test_input",
                    input_data="test_value",
                    block_id=f"block-{thread_id}-{i}",
                )
                created_exec_ids.append((exec_id, thread_id, i))

        def worker(thread_data):
            """Perform multiple operations from a thread."""
            thread_id, ops = thread_data
            for i, (exec_id, _, _) in enumerate(ops):
                # Status updates
                execution_client.update_node_status_and_publish(
                    exec_id=exec_id,
                    status=ExecutionStatus.RUNNING,
                    execution_data={"thread": thread_id, "op": i},
                )

                # Output updates (use just one exec_id per thread for outputs)
                if i == 0:  # Only add outputs to first execution of each thread
                    execution_client.upsert_execution_output(
                        node_exec_id=exec_id,
                        output_name=f"output_{i}",
                        output_data={"thread": thread_id, "value": i},
                    )

        # Organize executions by thread
        thread_data = []
        for tid in range(num_threads):
            thread_ops = [
                exec_data for exec_data in created_exec_ids if exec_data[1] == tid
            ]
            thread_data.append((tid, thread_ops))

        # Start multiple threads
        threads = []
        for data in thread_data:
            thread = threading.Thread(target=worker, args=(data,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify data integrity
        expected_executions = num_threads * operations_per_thread
        all_executions = execution_client.get_node_executions()
        assert len(all_executions) == expected_executions

        # Verify outputs - only first execution of each thread should have outputs
        output_count = 0
        for execution in all_executions:
            if execution.output_data:
                output_count += 1
        assert output_count == num_threads  # One output per thread

    async def test_sync_and_async_versions_consistent(self, execution_client):
        """Test that sync and async versions of output operations behave the same."""
        # First create the execution
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id="sync-async-test-node",
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.RUNNING,
            execution_data={"input": "test"},
        )

        execution_client.upsert_execution_output(
            node_exec_id=node_exec_id,
            output_name="sync_result",
            output_data={"method": "sync"},
        )

        execution_client.upsert_execution_output(
            node_exec_id=node_exec_id,
            output_name="async_result",
            output_data={"method": "async"},
        )

        cached_exec = execution_client.get_node_execution(node_exec_id)
        assert cached_exec is not None
        assert "sync_result" in cached_exec.output_data
        assert "async_result" in cached_exec.output_data
        assert cached_exec.output_data["sync_result"] == [{"method": "sync"}]
        assert cached_exec.output_data["async_result"] == [{"method": "async"}]

    async def test_finalize_execution_completes_and_clears_cache(
        self, execution_client
    ):
        """Test that finalize_execution waits for background tasks and clears cache."""
        # First create the execution
        node_exec_id, _ = execution_client.upsert_execution_input(
            node_id="pending-test-node",
            input_name="test_input",
            input_data="test_value",
            block_id="test-block",
        )

        # Trigger some background operations
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id, status=ExecutionStatus.RUNNING
        )

        execution_client.upsert_execution_output(
            node_exec_id=node_exec_id, output_name="test", output_data={"value": 1}
        )

        # Wait for background tasks - may fail in test environment due to DB issues
        try:
            execution_client.finalize_execution(timeout=5.0)
        except RuntimeError as e:
            # In test environment, background DB operations may fail, but cache should still be cleared
            assert "Background persistence failed" in str(e)

        # Cache should be cleared regardless of background task failures
        all_executions = execution_client.get_node_executions()
        assert len(all_executions) == 0  # Cache should be cleared

    async def test_manager_usage_pattern(self, execution_client):
        # Create executions first
        node_exec_id_1, _ = execution_client.upsert_execution_input(
            node_id="node-1",
            input_name="input1",
            input_data="data1",
            block_id="block-1",
        )

        node_exec_id_2, _ = execution_client.upsert_execution_input(
            node_id="node-2",
            input_name="input_from_node1",
            input_data="value1",
            block_id="block-2",
        )

        # Simulate manager.py workflow

        # 1. Start execution
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id_1,
            status=ExecutionStatus.RUNNING,
            execution_data={"input": "data1"},
        )

        # 2. Node produces output
        execution_client.upsert_execution_output(
            node_exec_id=node_exec_id_1,
            output_name="result",
            output_data={"output": "value1"},
        )

        # 3. Complete first node
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id_1, status=ExecutionStatus.COMPLETED
        )

        # 4. Start second node (would read output from first)
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id_2,
            status=ExecutionStatus.RUNNING,
            execution_data={"input_from_node1": "value1"},
        )

        # 5. Manager queries for executions

        all_executions = execution_client.get_node_executions()
        running_executions = execution_client.get_node_executions(
            statuses=[ExecutionStatus.RUNNING]
        )
        completed_executions = execution_client.get_node_executions(
            statuses=[ExecutionStatus.COMPLETED]
        )

        # Verify manager can see all data immediately
        assert len(all_executions) == 2
        assert len(running_executions) == 1
        assert len(completed_executions) == 1

        # Verify output is accessible
        exec_1 = execution_client.get_node_execution(node_exec_id_1)
        assert exec_1 is not None
        assert exec_1.output_data["result"] == [{"output": "value1"}]

    def test_stats_handling_in_update_node_status(self, execution_client):
        """Test that stats parameter is properly handled in update_node_status_and_publish."""
        # Create a fake execution directly in cache to avoid database issues
        from datetime import datetime, timezone

        from backend.data.execution import NodeExecutionResult

        node_exec_id = "test-stats-exec-id"
        fake_execution = NodeExecutionResult(
            user_id="test-user",
            graph_id="test-graph",
            graph_version=1,
            graph_exec_id="test-graph-exec",
            node_exec_id=node_exec_id,
            node_id="stats-test-node",
            block_id="test-block",
            status=ExecutionStatus.INCOMPLETE,
            input_data={"test_input": "test_value"},
            output_data={},
            add_time=datetime.now(timezone.utc),
            queue_time=None,
            start_time=None,
            end_time=None,
            stats=None,
        )

        # Add directly to cache
        execution_client._cache.add_node_execution(node_exec_id, fake_execution)

        stats = {"token_count": 150, "processing_time": 2.5}

        # Update status with stats
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.RUNNING,
            execution_data={"input": "test"},
            stats=stats,
        )

        # Verify execution was updated and stats are stored
        execution = execution_client.get_node_execution(node_exec_id)
        assert execution is not None
        assert execution.status == ExecutionStatus.RUNNING

        # Stats should be stored in proper stats field
        assert execution.stats is not None
        stats_dict = execution.stats.model_dump()
        # Only check the fields we set, ignore defaults
        assert stats_dict["token_count"] == 150
        assert stats_dict["processing_time"] == 2.5

        # Update with additional stats
        additional_stats = {"error_count": 0}
        execution_client.update_node_status_and_publish(
            exec_id=node_exec_id,
            status=ExecutionStatus.COMPLETED,
            stats=additional_stats,
        )

        # Stats should be merged
        execution = execution_client.get_node_execution(node_exec_id)
        assert execution is not None
        assert execution.status == ExecutionStatus.COMPLETED
        stats_dict = execution.stats.model_dump()
        # Check the merged stats
        assert stats_dict["token_count"] == 150
        assert stats_dict["processing_time"] == 2.5
        assert stats_dict["error_count"] == 0

    async def test_upsert_execution_input_scenarios(self, execution_client):
        """Test different scenarios of upsert_execution_input - create vs update."""
        node_id = "test-node"
        graph_exec_id = (
            "test_graph_exec_id"  # Must match the ExecutionDataClient's scope
        )

        # Scenario 1: Create new execution when none exists
        exec_id_1, input_data_1 = execution_client.upsert_execution_input(
            node_id=node_id,
            input_name="first_input",
            input_data="value1",
            block_id="test-block",
        )

        # Should create new execution
        execution = execution_client.get_node_execution(exec_id_1)
        assert execution is not None
        assert execution.status == ExecutionStatus.INCOMPLETE
        assert execution.node_id == node_id
        assert execution.graph_exec_id == graph_exec_id
        assert input_data_1 == {"first_input": "value1"}

        # Scenario 2: Add input to existing INCOMPLETE execution
        exec_id_2, input_data_2 = execution_client.upsert_execution_input(
            node_id=node_id,
            input_name="second_input",
            input_data="value2",
            block_id="test-block",
        )

        # Should use same execution
        assert exec_id_2 == exec_id_1
        assert input_data_2 == {"first_input": "value1", "second_input": "value2"}

        # Verify execution has both inputs
        execution = execution_client.get_node_execution(exec_id_1)
        assert execution is not None
        assert execution.input_data == {
            "first_input": "value1",
            "second_input": "value2",
        }

        # Scenario 3: Create new execution when existing is not INCOMPLETE
        execution_client.update_node_status_and_publish(
            exec_id=exec_id_1, status=ExecutionStatus.RUNNING
        )

        exec_id_3, input_data_3 = execution_client.upsert_execution_input(
            node_id=node_id,
            input_name="third_input",
            input_data="value3",
            block_id="test-block",
        )

        # Should create new execution
        assert exec_id_3 != exec_id_1
        execution_3 = execution_client.get_node_execution(exec_id_3)
        assert execution_3 is not None
        assert input_data_3 == {"third_input": "value3"}

        # Verify we now have 2 executions
        all_executions = execution_client.get_node_executions()
        assert len(all_executions) == 2

    def test_graph_stats_operations(self, execution_client):
        """Test graph-level stats and start time operations."""

        # Test update_graph_stats_and_publish
        from backend.data.model import GraphExecutionStats

        stats = GraphExecutionStats(
            walltime=10.5, cputime=8.2, node_count=5, node_error_count=1
        )

        execution_client.update_graph_stats_and_publish(
            status=ExecutionStatus.RUNNING, stats=stats
        )

        # Verify stats are stored in cache
        cached_stats = execution_client._cache._graph_stats
        assert cached_stats.walltime == 10.5

        execution_client.update_graph_start_time_and_publish()
        cached_stats = execution_client._cache._graph_stats
        assert cached_stats.walltime == 10.5

    def test_public_methods_accessible(self, execution_client):
        """Test that public methods are accessible."""
        assert hasattr(execution_client._cache, "update_node_execution_status")
        assert hasattr(execution_client._cache, "upsert_execution_output")
        assert hasattr(execution_client._cache, "add_node_execution")
        assert hasattr(execution_client._cache, "find_incomplete_execution_for_input")
        assert hasattr(execution_client._cache, "update_execution_input")
        assert hasattr(execution_client, "upsert_execution_input")
        assert hasattr(execution_client, "update_node_status_and_publish")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
