import unittest
from unittest.mock import MagicMock

from backend.executor.cached_client import wrap_client
from backend.executor.simple_cache import clear_cache, get_cache


class TestExecutionCache(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.mock_client = MagicMock()
        self.cached_client = wrap_client(self.mock_client)

    def test_node_caching(self):
        self.mock_client.get_node.return_value = {"id": "node_1", "data": "test"}

        # First call should hit backend
        result1 = self.cached_client.get_node("node_1")
        self.assertEqual(self.mock_client.get_node.call_count, 1)

        # Second call should use cache
        result2 = self.cached_client.get_node("node_1")
        self.assertEqual(self.mock_client.get_node.call_count, 1)
        self.assertEqual(result1, result2)

    def test_node_executions_caching(self):
        self.mock_client.get_node_executions.return_value = [
            {"id": "exec_1", "status": "completed"}
        ]

        # First call should hit backend
        result1 = self.cached_client.get_node_executions("graph_1")
        self.assertEqual(self.mock_client.get_node_executions.call_count, 1)

        # Second call should use cache
        result2 = self.cached_client.get_node_executions("graph_1")
        self.assertEqual(self.mock_client.get_node_executions.call_count, 1)
        self.assertEqual(result1, result2)

    def test_output_updates_queued(self):
        # Should not call backend immediately
        self.cached_client.upsert_execution_output("exec_1", {"data": "output"})
        self.mock_client.upsert_execution_output.assert_not_called()

        # Check that it was queued
        cache = get_cache()
        outputs, _ = cache.get_pending_updates()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["node_exec_id"], "exec_1")

    def test_status_updates_queued(self):
        # Should not call backend immediately
        self.cached_client.update_node_execution_status("exec_1", "completed")
        self.mock_client.update_node_execution_status.assert_not_called()

        # Check that it was queued
        cache = get_cache()
        _, statuses = cache.get_pending_updates()
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["node_exec_id"], "exec_1")


if __name__ == "__main__":
    unittest.main()
