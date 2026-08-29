"""Tests for ExecutionQueue thread-safety."""

import queue
import threading

from backend.data.execution import ExecutionQueue


def test_execution_queue_uses_stdlib_queue():
    """Verify ExecutionQueue uses queue.Queue (not multiprocessing)."""
    q = ExecutionQueue()
    assert isinstance(q.queue, queue.Queue)


def test_basic_operations():
    """Test add, get, empty, and get_or_none."""
    q = ExecutionQueue()

    assert q.empty() is True
    assert q.get_or_none() is None

    result = q.add("item1")
    assert result == "item1"
    assert q.empty() is False

    item = q.get()
    assert item == "item1"
    assert q.empty() is True


def test_thread_safety():
    """Test concurrent access from multiple threads."""
    q = ExecutionQueue()
    results = []
    num_items = 100

    def producer():
        for i in range(num_items):
            q.add(f"item_{i}")

    def consumer():
        count = 0
        while count < num_items:
            item = q.get_or_none()
            if item is not None:
                results.append(item)
                count += 1

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join(timeout=5)
    consumer_thread.join(timeout=5)

    assert len(results) == num_items
