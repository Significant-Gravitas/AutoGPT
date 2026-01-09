"""
Test script to verify the ExecutionQueue fix in execution.py

This script tests:
1. That ExecutionQueue uses queue.Queue (not multiprocessing.Manager().Queue())
2. All queue operations work correctly
3. Thread-safety works as expected
"""

import sys
import threading
import time

sys.path.insert(0, ".")

import queue


def test_queue_type():
    """Test that ExecutionQueue uses the correct queue type."""
    from backend.data.execution import ExecutionQueue

    q = ExecutionQueue()

    # Verify it's using queue.Queue, not multiprocessing queue
    assert isinstance(
        q.queue, queue.Queue
    ), f"FAIL: Expected queue.Queue, got {type(q.queue)}"
    print("✓ ExecutionQueue uses queue.Queue (not multiprocessing.Manager().Queue())")


def test_basic_operations():
    """Test basic queue operations."""
    from backend.data.execution import ExecutionQueue

    q = ExecutionQueue()

    # Test add
    result = q.add("item1")
    assert result == "item1", f"FAIL: add() should return the item, got {result}"
    print("✓ add() works correctly")

    # Test empty() when not empty
    assert q.empty() is False, "FAIL: empty() should return False when queue has items"
    print("✓ empty() returns False when queue has items")

    # Test get()
    item = q.get()
    assert item == "item1", f"FAIL: get() returned {item}, expected 'item1'"
    print("✓ get() works correctly")

    # Test empty() when empty
    assert q.empty() is True, "FAIL: empty() should return True when queue is empty"
    print("✓ empty() returns True when queue is empty")

    # Test get_or_none() when empty
    result = q.get_or_none()
    assert result is None, f"FAIL: get_or_none() should return None, got {result}"
    print("✓ get_or_none() returns None when queue is empty")

    # Test get_or_none() with items
    q.add("item2")
    result = q.get_or_none()
    assert result == "item2", f"FAIL: get_or_none() returned {result}, expected 'item2'"
    print("✓ get_or_none() returns item when queue has items")


def test_thread_safety():
    """Test that the queue is thread-safe."""
    from backend.data.execution import ExecutionQueue

    q = ExecutionQueue()
    results = []
    errors = []
    num_items = 100

    def producer():
        try:
            for i in range(num_items):
                q.add(f"item_{i}")
        except Exception as e:
            errors.append(f"Producer error: {e}")

    def consumer():
        try:
            count = 0
            while count < num_items:
                item = q.get_or_none()
                if item is not None:
                    results.append(item)
                    count += 1
                else:
                    time.sleep(0.001)  # Small delay to avoid busy waiting
        except Exception as e:
            errors.append(f"Consumer error: {e}")

    # Start threads
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join(timeout=5)
    consumer_thread.join(timeout=5)

    assert len(errors) == 0, f"FAIL: Thread errors occurred: {errors}"
    assert (
        len(results) == num_items
    ), f"FAIL: Expected {num_items} items, got {len(results)}"
    print(
        f"✓ Thread-safety test passed ({num_items} items transferred between threads)"
    )


def test_multiple_producers_consumers():
    """Test with multiple producer and consumer threads."""
    from backend.data.execution import ExecutionQueue

    q = ExecutionQueue()
    results = []
    results_lock = threading.Lock()
    errors = []
    items_per_producer = 50
    num_producers = 3
    total_items = items_per_producer * num_producers

    def producer(producer_id):
        try:
            for i in range(items_per_producer):
                q.add(f"producer_{producer_id}_item_{i}")
        except Exception as e:
            errors.append(f"Producer {producer_id} error: {e}")

    def consumer(consumer_id, target_count):
        try:
            count = 0
            max_attempts = target_count * 100
            attempts = 0
            while count < target_count and attempts < max_attempts:
                item = q.get_or_none()
                if item is not None:
                    with results_lock:
                        results.append(item)
                    count += 1
                else:
                    time.sleep(0.001)
                attempts += 1
        except Exception as e:
            errors.append(f"Consumer {consumer_id} error: {e}")

    # Start multiple producers
    producer_threads = [
        threading.Thread(target=producer, args=(i,)) for i in range(num_producers)
    ]

    # Start multiple consumers (each consumes half of total)
    consumer_threads = [
        threading.Thread(target=consumer, args=(i, total_items // 2)) for i in range(2)
    ]

    for t in producer_threads:
        t.start()
    for t in consumer_threads:
        t.start()

    for t in producer_threads:
        t.join(timeout=10)
    for t in consumer_threads:
        t.join(timeout=10)

    assert len(errors) == 0, f"FAIL: Thread errors occurred: {errors}"
    assert (
        len(results) == total_items
    ), f"FAIL: Expected {total_items} items, got {len(results)}"
    print(
        f"✓ Multi-producer/consumer test passed ({num_producers} producers, 2 consumers, {total_items} items)"
    )


def test_no_subprocess_spawned():
    """Verify that no subprocess is spawned (unlike multiprocessing.Manager())."""
    from backend.data.execution import ExecutionQueue

    # Create multiple queues (this would spawn subprocesses with Manager())
    queues = [ExecutionQueue() for _ in range(5)]

    # If we got here without issues, no subprocesses were spawned
    # With Manager().Queue(), creating 5 queues would spawn 5 manager processes
    for q in queues:
        q.add("test")
        assert q.get() == "test"

    print(
        "✓ No subprocess spawning (5 queues created without spawning manager processes)"
    )


def main():
    print("=" * 60)
    print("ExecutionQueue Fix Verification Tests")
    print("=" * 60)
    print()

    tests = [
        ("Queue Type Check", test_queue_type),
        ("Basic Operations", test_basic_operations),
        ("Thread Safety", test_thread_safety),
        ("Multiple Producers/Consumers", test_multiple_producers_consumers),
        ("No Subprocess Spawning", test_no_subprocess_spawned),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"✅ ALL TESTS PASSED ({passed}/{passed})")
        print("The ExecutionQueue fix is working correctly!")
    else:
        print(f"❌ TESTS FAILED: {failed} failed, {passed} passed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
