import threading
import time
import unittest
from unittest.mock import MagicMock

from backend.executor.cached_client import wrap_client
from backend.executor.simple_cache import SimpleExecutorCache, clear_cache, get_cache


class CachePerformanceTest(unittest.TestCase):
    """
    Test suite for executor cache performance optimizations.
    Tests the caching functionality that reduces blocking I/O operations.
    """

    def setUp(self):
        clear_cache()

    def test_basic_cache_functionality(self):
        """Test basic cache operations work correctly"""
        cache = SimpleExecutorCache()

        # Test node caching
        test_node = {"id": "node_1", "data": "test_data"}
        cache.cache_node("node_1", test_node)
        retrieved = cache.get_node("node_1")
        self.assertEqual(retrieved, test_node)

        # Test node executions caching
        test_executions = [{"id": "exec_1", "status": "completed"}]
        cache.cache_node_executions("graph_1", test_executions)
        retrieved_execs = cache.get_node_executions("graph_1")
        self.assertEqual(retrieved_execs, test_executions)

    def test_queue_functionality(self):
        """Test output and status queuing for non-blocking operations"""
        cache = SimpleExecutorCache()

        # Queue updates
        cache.queue_output_update("exec_1", {"data": "output_1"})
        cache.queue_status_update("exec_1", "completed")

        # Get pending updates
        outputs, statuses = cache.get_pending_updates()

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(statuses), 1)
        self.assertEqual(outputs[0]["node_exec_id"], "exec_1")
        self.assertEqual(statuses[0]["node_exec_id"], "exec_1")

        # Queue should be empty after retrieval
        outputs2, statuses2 = cache.get_pending_updates()
        self.assertEqual(len(outputs2), 0)
        self.assertEqual(len(statuses2), 0)

    def test_thread_safety(self):
        """Test cache is thread-safe under concurrent operations"""
        cache = SimpleExecutorCache()

        def worker(worker_id):
            for i in range(10):
                cache.cache_node(
                    f"node_{worker_id}_{i}", {"worker": worker_id, "item": i}
                )
                cache.queue_output_update(
                    f"exec_{worker_id}_{i}", {"worker": worker_id}
                )
                cache.queue_status_update(f"exec_{worker_id}_{i}", "completed")

        # Run concurrent operations
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no data corruption
        outputs, statuses = cache.get_pending_updates()
        self.assertEqual(len(outputs), 50)  # 5 workers * 10 items
        self.assertEqual(len(statuses), 50)

    def test_cached_client_reduces_calls(self):
        """Test cached client wrapper reduces backend calls"""
        mock_client = MagicMock()
        mock_client.get_node.return_value = {"id": "test_node", "data": "test"}
        mock_client.get_node_executions.return_value = [
            {"id": "exec_1", "status": "completed"}
        ]

        clear_cache()
        cached_client = wrap_client(mock_client)

        # First calls should hit backend
        result1 = cached_client.get_node("test_node")
        exec1 = cached_client.get_node_executions("graph_1")
        self.assertEqual(mock_client.get_node.call_count, 1)
        self.assertEqual(mock_client.get_node_executions.call_count, 1)

        # Second calls should hit cache
        result2 = cached_client.get_node("test_node")
        exec2 = cached_client.get_node_executions("graph_1")
        self.assertEqual(mock_client.get_node.call_count, 1)  # No increase
        self.assertEqual(mock_client.get_node_executions.call_count, 1)  # No increase

        # Results should be identical
        self.assertEqual(result1, result2)
        self.assertEqual(exec1, exec2)

    def test_non_blocking_operations(self):
        """Test operations that should be non-blocking return immediately"""
        mock_client = MagicMock()
        cached_client = wrap_client(mock_client)

        # These should return immediately without calling backend
        result1 = cached_client.upsert_execution_output("exec_1", {"data": "output"})
        result2 = cached_client.update_node_execution_status("exec_1", "completed")

        self.assertEqual(result1, {"success": True})
        self.assertEqual(result2, {"success": True})
        mock_client.upsert_execution_output.assert_not_called()
        mock_client.update_node_execution_status.assert_not_called()

        # Verify they were queued
        cache = get_cache()
        outputs, statuses = cache.get_pending_updates()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(statuses), 1)

    def test_performance_improvement(self):
        """Test that caching provides measurable performance improvement"""

        class SlowMockClient:
            def __init__(self):
                self.call_count = 0

            def get_node(self, node_id):
                self.call_count += 1
                time.sleep(0.01)  # Simulate 10ms I/O delay
                return {"id": node_id, "data": "test"}

        clear_cache()
        slow_client = SlowMockClient()
        cached_client = wrap_client(slow_client)

        # Time first call (should be slow due to I/O)
        start = time.time()
        cached_client.get_node("perf_test")
        time1 = time.time() - start

        # Time second call (should be fast due to cache)
        start = time.time()
        cached_client.get_node("perf_test")
        time2 = time.time() - start

        # Verify performance improvement
        self.assertGreater(time1, 0.01)  # First call should be slow (>10ms)
        self.assertLess(time2, 0.005)  # Second call should be fast (<5ms)
        self.assertEqual(slow_client.call_count, 1)  # Backend called only once

        speedup = time1 / time2 if time2 > 0 else float("inf")
        print(
            f"Cache speedup: {speedup:.1f}x (first: {time1*1000:.1f}ms, cached: {time2*1000:.1f}ms)"
        )


if __name__ == "__main__":
    unittest.main()
