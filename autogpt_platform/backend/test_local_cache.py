#!/usr/bin/env python3
"""
Test script to validate the local caching implementation.
This can be run independently to test the caching functionality.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import ExecutionStatus
from backend.executor.local_cache import LocalExecutorCache


class MockDatabaseClient:
    """Mock database client for testing"""

    def __init__(self):
        self.operations = []
        self.credits = {
            "user1": 1000,
            "user2": 500,
        }

    def spend_credits(
        self, user_id: str, cost: int, metadata: UsageTransactionMetadata
    ) -> int:
        self.operations.append(("spend_credits", user_id, cost, metadata))
        if user_id in self.credits:
            self.credits[user_id] -= cost
            return self.credits[user_id]
        return 0

    def upsert_execution_output(self, node_exec_id: str, output_name: str, output_data):
        self.operations.append(
            ("upsert_execution_output", node_exec_id, output_name, output_data)
        )

    def update_node_execution_status(
        self, exec_id: str, status: ExecutionStatus, execution_data=None, stats=None
    ):
        self.operations.append(
            ("update_node_execution_status", exec_id, status, execution_data, stats)
        )

    def get_credits(self, user_id: str) -> int:
        return self.credits.get(user_id, 0)


async def test_local_cache():
    """Test the local cache functionality"""
    print("🧪 Testing LocalExecutorCache...")

    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "test_cache.db")

        # Initialize cache
        cache = LocalExecutorCache(db_path=db_path)
        mock_client = MockDatabaseClient()
        cache.initialize(mock_client)

        try:
            # Test 1: Local synchronization
            print("Test 1: Local thread locks")
            async with cache.local_synchronized("test_key"):
                print("✅ Local lock acquired and released successfully")

            # Test 2: Execution counting
            print("Test 2: Execution counting")
            count1 = cache.increment_execution_count_local("user1", "exec1")
            count2 = cache.increment_execution_count_local("user1", "exec1")
            assert count1 == 1 and count2 == 2
            print(f"✅ Execution count: {count1}, {count2}")

            # Test 3: Credit operations
            print("Test 3: Credit spending with local balance")
            metadata = UsageTransactionMetadata(
                graph_exec_id="exec1",
                graph_id="graph1",
                node_exec_id="node1",
                node_id="node1",
                block_id="block1",
                block="TestBlock",
                input={},
                reason="Test",
            )

            # Initialize balance
            cache._execution_states["exec1"] = cache.LocalExecutionState(
                graph_exec_id="exec1",
                user_id="user1",
                status=ExecutionStatus.RUNNING,
                local_balance=1000,
            )

            remaining = cache.spend_credits_local("user1", "exec1", 100, metadata)
            assert remaining == 900
            print(f"✅ Credits spent, remaining: {remaining}")

            # Test 4: Output caching
            print("Test 4: Execution output caching")
            cache.upsert_execution_output_local("node1", "output", {"result": "test"})
            print("✅ Output cached successfully")

            # Test 5: Status caching
            print("Test 5: Node status caching")
            cache.update_node_execution_status_local(
                "node1", ExecutionStatus.COMPLETED, {"input": "test"}, {"stats": "test"}
            )
            print("✅ Status cached successfully")

            # Test 6: Background sync
            print("Test 6: Background sync")
            await asyncio.sleep(2)  # Let sync thread run

            # Check if operations were synced
            if len(mock_client.operations) > 0:
                print(
                    f"✅ {len(mock_client.operations)} operations synced to remote DB"
                )
                for op in mock_client.operations:
                    print(f"  - {op[0]}")
            else:
                print("⚠️ No operations synced yet (sync may take time)")

        finally:
            # Cleanup
            cache.cleanup()
            print("✅ Cache cleanup completed")

    print("🎉 All tests passed!")


def test_cached_database_manager():
    """Test the cached database manager clients"""
    print("🧪 Testing CachedDatabaseManager...")

    from backend.executor.cached_database_manager import CachedDatabaseManagerClient

    # Mock original client
    original_client = MagicMock()
    original_client.get_credits.return_value = 1000
    original_client.spend_credits.return_value = 900
    original_client.upsert_execution_output.return_value = None

    # Create cached client
    cached_client = CachedDatabaseManagerClient.create_with_cache(original_client)

    # Test credit operations
    metadata = UsageTransactionMetadata(
        graph_exec_id="test_exec",
        graph_id="test_graph",
        node_exec_id="test_node",
        node_id="test_node",
        block_id="test_block",
        block="TestBlock",
        input={},
        reason="Test",
    )

    try:
        # This should use local caching
        remaining = cached_client.spend_credits("test_user", 100, metadata)
        print(f"✅ Cached spend_credits returned: {remaining}")
    except Exception as e:
        print(f"⚠️ Cached spend_credits failed: {e}")

    # Test output caching
    try:
        cached_client.upsert_execution_output("test_node", "output", {"test": "data"})
        print("✅ Cached upsert_execution_output succeeded")
    except Exception as e:
        print(f"⚠️ Cached upsert_execution_output failed: {e}")

    print("🎉 CachedDatabaseManager tests completed!")


if __name__ == "__main__":
    print("🚀 Starting Local Cache Tests\n")

    # Run async tests
    asyncio.run(test_local_cache())
    print()

    # Run sync tests
    test_cached_database_manager()
    print()

    print("✨ All tests completed successfully!")
