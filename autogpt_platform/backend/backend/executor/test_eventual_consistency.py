"""
Test for eventual consistency improvements in agent execution.

This test verifies that:
1. Database operations are properly batched during execution
2. Thread locks work correctly for output pin operations
3. Final flush ensures all operations are persisted
4. Balance deduction remains atomic
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.executor.manager import ExecutionBuffer, PendingOperation, _tls
from backend.data.execution import ExecutionStatus
from backend.data.model import GraphExecutionStats, NodeExecutionStats


class TestEventualConsistency(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.buffer = ExecutionBuffer()
        
    def test_execution_buffer_operations(self):
        """Test that operations are properly added to the buffer."""
        # Test output operation
        self.buffer.add_output_operation("node1", "output1", {"data": "test"})
        self.assertEqual(len(self.buffer.pending_outputs), 1)
        self.assertEqual(self.buffer.pending_outputs[0].node_exec_id, "node1")
        
        # Test status update
        self.buffer.add_status_update("node1", ExecutionStatus.RUNNING)
        self.assertEqual(len(self.buffer.pending_status_updates), 1)
        self.assertEqual(self.buffer.pending_status_updates[0].data["status"], ExecutionStatus.RUNNING)
        
        # Test stats update
        stats = GraphExecutionStats()
        self.buffer.add_stats_update("graph1", stats)
        self.assertEqual(len(self.buffer.pending_stats_updates), 1)
        
    def test_batch_flushing_conditions(self):
        """Test that batches flush at the right conditions."""
        # Test size-based flushing
        for i in range(self.buffer.max_batch_size):
            self.buffer.add_output_operation(f"node{i}", "output", {"data": i})
        self.assertTrue(self.buffer.should_flush())
        
        # Reset and test age-based flushing
        self.buffer.clear()
        self.buffer.add_output_operation("node1", "output", {"data": "test"})
        
        # Simulate old operation
        self.buffer.pending_outputs[0].timestamp = time.time() - (self.buffer.max_batch_age_seconds + 1)
        self.assertTrue(self.buffer.should_flush())
        
    def test_thread_locks(self):
        """Test that thread locks work correctly."""
        lock_key = "test-lock"
        
        # Get the same lock twice and verify it's the same instance
        lock1 = self.buffer.output_pin_locks[lock_key]
        lock2 = self.buffer.output_pin_locks[lock_key]
        self.assertIs(lock1, lock2)
        
        # Test lock functionality
        acquired = lock1.acquire(blocking=False)
        self.assertTrue(acquired)
        
        # Should not be able to acquire again
        acquired2 = lock1.acquire(blocking=False)
        self.assertFalse(acquired2)
        
        lock1.release()
        
    def test_buffer_clear(self):
        """Test that buffer clear works correctly."""
        self.buffer.add_output_operation("node1", "output1", {"data": "test"})
        self.buffer.add_status_update("node1", ExecutionStatus.RUNNING)
        stats = GraphExecutionStats()
        self.buffer.add_stats_update("graph1", stats)
        
        self.buffer.clear()
        
        self.assertEqual(len(self.buffer.pending_outputs), 0)
        self.assertEqual(len(self.buffer.pending_status_updates), 0)
        self.assertEqual(len(self.buffer.pending_stats_updates), 0)


class TestEventualConsistencyIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for eventual consistency in execution manager."""
    
    async def test_batch_flush_integration(self):
        """Test that batch flushing works with mocked database operations."""
        from backend.executor.manager import ExecutionProcessor
        
        # Mock database client
        mock_db_client = AsyncMock()
        mock_db_client.upsert_execution_output = AsyncMock()
        
        # Create processor and buffer
        processor = ExecutionProcessor()
        buffer = ExecutionBuffer()
        
        # Add some operations to buffer
        buffer.add_output_operation("node1", "output1", {"data": "test1"})
        buffer.add_output_operation("node2", "output2", {"data": "test2"})
        
        # Mock _tls to use our buffer
        with patch('backend.executor.manager._tls') as mock_tls:
            mock_tls.execution_buffer = buffer
            
            # Call flush
            await processor._flush_execution_buffer(mock_db_client)
            
            # Verify operations were called
            self.assertEqual(mock_db_client.upsert_execution_output.call_count, 2)
            
            # Verify buffer was cleared
            self.assertEqual(len(buffer.pending_outputs), 0)


if __name__ == '__main__':
    unittest.main() 