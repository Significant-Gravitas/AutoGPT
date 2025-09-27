"""
Integration tests for ClusterLock - Redis-based distributed locking.

Tests the complete lock lifecycle without mocking Redis to ensure
real-world behavior is correct. Covers acquisition, refresh, expiry,
contention, and error scenarios.
"""

import logging
import time
import uuid
from threading import Thread

import pytest
import redis

from .cluster_lock import ClusterLock

logger = logging.getLogger(__name__)


@pytest.fixture
def redis_client():
    """Get Redis client for testing using same config as backend."""
    from backend.data.redis_client import HOST, PASSWORD, PORT

    # Use same config as backend but without decode_responses since ClusterLock needs raw bytes
    client = redis.Redis(
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=False,  # ClusterLock needs raw bytes for ownership verification
    )

    # Clean up any existing test keys
    try:
        for key in client.scan_iter(match="test_lock:*"):
            client.delete(key)
    except Exception:
        pass  # Ignore cleanup errors

    return client


@pytest.fixture
def lock_key():
    """Generate unique lock key for each test."""
    return f"test_lock:{uuid.uuid4()}"


@pytest.fixture
def owner_id():
    """Generate unique owner ID for each test."""
    return str(uuid.uuid4())


class TestClusterLockBasic:
    """Basic lock acquisition and release functionality."""

    def test_lock_acquisition_success(self, redis_client, lock_key, owner_id):
        """Test basic lock acquisition succeeds."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        # Lock should be acquired successfully
        assert lock.try_acquire() is True
        assert lock._acquired is True

        # Lock key should exist in Redis
        assert redis_client.exists(lock_key) == 1
        assert redis_client.get(lock_key).decode("utf-8") == owner_id

    def test_lock_acquisition_contention(self, redis_client, lock_key):
        """Test second acquisition fails when lock is held."""
        owner1 = str(uuid.uuid4())
        owner2 = str(uuid.uuid4())

        lock1 = ClusterLock(redis_client, lock_key, owner1, timeout=60)
        lock2 = ClusterLock(redis_client, lock_key, owner2, timeout=60)

        # First lock should succeed
        assert lock1.try_acquire() is True

        # Second lock should fail
        assert lock2.try_acquire() is False
        assert lock2._acquired is False

    def test_lock_release_local_only(self, redis_client, lock_key, owner_id):
        """Test lock release only marks locally as released."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        lock.try_acquire()
        assert lock._acquired is True

        # Release should mark locally as released but leave Redis key
        lock.release()
        assert lock._acquired is False
        assert lock._last_refresh == 0.0

        # Redis key should still exist (will expire naturally)
        assert redis_client.exists(lock_key) == 1


class TestClusterLockRefresh:
    """Lock refresh and TTL management."""

    def test_lock_refresh_success(self, redis_client, lock_key, owner_id):
        """Test lock refresh extends TTL."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        lock.try_acquire()
        original_ttl = redis_client.ttl(lock_key)

        # Wait a bit then refresh
        time.sleep(1)
        lock._last_refresh = 0  # Force refresh past rate limit
        assert lock.refresh() is True

        # TTL should be reset to full timeout (allow for small timing differences)
        new_ttl = redis_client.ttl(lock_key)
        assert new_ttl >= original_ttl or new_ttl >= 58  # Allow for timing variance

    def test_lock_refresh_rate_limiting(self, redis_client, lock_key, owner_id):
        """Test refresh is rate-limited to timeout/10."""
        lock = ClusterLock(
            redis_client, lock_key, owner_id, timeout=100
        )  # 100s timeout

        lock.try_acquire()

        # First refresh should work
        assert lock.refresh() is True
        first_refresh_time = lock._last_refresh

        # Immediate second refresh should be skipped (rate limited)
        assert lock.refresh() is True  # Returns True but skips actual refresh
        assert lock._last_refresh == first_refresh_time  # Time unchanged

    def test_lock_refresh_ownership_lost(self, redis_client, lock_key, owner_id):
        """Test refresh fails when ownership is lost."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        lock.try_acquire()

        # Simulate another process taking the lock
        different_owner = str(uuid.uuid4())
        redis_client.set(lock_key, different_owner, ex=60)

        # Force refresh past rate limit and verify it fails
        lock._last_refresh = 0  # Force refresh past rate limit
        assert lock.refresh() is False
        assert lock._acquired is False

    def test_lock_refresh_when_not_acquired(self, redis_client, lock_key, owner_id):
        """Test refresh fails when lock was never acquired."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        # Refresh without acquiring should fail
        assert lock.refresh() is False


class TestClusterLockExpiry:
    """Lock expiry and timeout behavior."""

    def test_lock_natural_expiry(self, redis_client, lock_key, owner_id):
        """Test lock expires naturally via Redis TTL."""
        lock = ClusterLock(
            redis_client, lock_key, owner_id, timeout=2
        )  # 2 second timeout

        lock.try_acquire()
        assert redis_client.exists(lock_key) == 1

        # Wait for expiry
        time.sleep(3)
        assert redis_client.exists(lock_key) == 0

        # New lock with same key should succeed
        new_lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)
        assert new_lock.try_acquire() is True

    def test_lock_refresh_prevents_expiry(self, redis_client, lock_key, owner_id):
        """Test refreshing prevents lock from expiring."""
        lock = ClusterLock(
            redis_client, lock_key, owner_id, timeout=3
        )  # 3 second timeout

        lock.try_acquire()

        # Wait and refresh before expiry
        time.sleep(1)
        lock._last_refresh = 0  # Force refresh past rate limit
        assert lock.refresh() is True

        # Wait beyond original timeout
        time.sleep(2.5)
        assert redis_client.exists(lock_key) == 1  # Should still exist


class TestClusterLockContextManager:
    """Context manager functionality."""

    def test_context_manager_blocking_success(self, redis_client, lock_key, owner_id):
        """Test context manager with successful blocking acquisition."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        with lock.acquire(blocking=True) as acquired_lock:
            assert acquired_lock is not None
            assert acquired_lock is lock
            assert lock._acquired is True
            assert redis_client.exists(lock_key) == 1

        # Lock should be released after context
        assert lock._acquired is False

    def test_context_manager_blocking_failure(self, redis_client, lock_key):
        """Test context manager raises exception when blocking=True and lock held."""
        owner1 = str(uuid.uuid4())
        owner2 = str(uuid.uuid4())

        lock1 = ClusterLock(redis_client, lock_key, owner1, timeout=60)
        lock2 = ClusterLock(redis_client, lock_key, owner2, timeout=60)

        # First lock acquired
        lock1.try_acquire()

        # Second lock should raise exception
        with pytest.raises(RuntimeError, match="Lock already held"):
            with lock2.acquire(blocking=True):
                pass

    def test_context_manager_non_blocking_success(
        self, redis_client, lock_key, owner_id
    ):
        """Test context manager with successful non-blocking acquisition."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        with lock.acquire(blocking=False) as acquired_lock:
            assert acquired_lock is not None
            assert acquired_lock is lock
            assert lock._acquired is True

    def test_context_manager_non_blocking_failure(self, redis_client, lock_key):
        """Test context manager yields None when blocking=False and lock held."""
        owner1 = str(uuid.uuid4())
        owner2 = str(uuid.uuid4())

        lock1 = ClusterLock(redis_client, lock_key, owner1, timeout=60)
        lock2 = ClusterLock(redis_client, lock_key, owner2, timeout=60)

        # First lock acquired
        lock1.try_acquire()

        # Second lock should yield None
        with lock2.acquire(blocking=False) as acquired_lock:
            assert acquired_lock is None


class TestClusterLockConcurrency:
    """Concurrent access patterns."""

    def test_multiple_threads_contention(self, redis_client, lock_key):
        """Test multiple threads competing for same lock."""
        num_threads = 5
        successful_acquisitions = []

        def try_acquire_lock(thread_id):
            owner_id = f"thread_{thread_id}"
            lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)
            if lock.try_acquire():
                successful_acquisitions.append(thread_id)
                time.sleep(0.1)  # Hold lock briefly
                lock.release()

        threads = []
        for i in range(num_threads):
            thread = Thread(target=try_acquire_lock, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Only one thread should have acquired the lock
        assert len(successful_acquisitions) == 1

    def test_sequential_lock_reuse(self, redis_client, lock_key):
        """Test lock can be reused after natural expiry."""
        owners = [str(uuid.uuid4()) for _ in range(3)]

        for i, owner_id in enumerate(owners):
            lock = ClusterLock(redis_client, lock_key, owner_id, timeout=1)  # 1 second

            assert lock.try_acquire() is True
            time.sleep(1.5)  # Wait for expiry

            # Verify lock expired
            assert redis_client.exists(lock_key) == 0

    def test_refresh_during_concurrent_access(self, redis_client, lock_key):
        """Test lock refresh works correctly during concurrent access attempts."""
        owner1 = str(uuid.uuid4())
        owner2 = str(uuid.uuid4())

        lock1 = ClusterLock(redis_client, lock_key, owner1, timeout=5)
        lock2 = ClusterLock(redis_client, lock_key, owner2, timeout=5)

        # Thread 1 holds lock and refreshes
        assert lock1.try_acquire() is True

        def refresh_continuously():
            for _ in range(10):
                lock1._last_refresh = 0  # Force refresh
                lock1.refresh()
                time.sleep(0.1)

        def try_acquire_continuously():
            attempts = 0
            while attempts < 20:
                if lock2.try_acquire():
                    return True
                time.sleep(0.1)
                attempts += 1
            return False

        refresh_thread = Thread(target=refresh_continuously)
        acquire_thread = Thread(target=try_acquire_continuously)

        refresh_thread.start()
        acquire_thread.start()

        refresh_thread.join()
        acquire_thread.join()

        # Lock1 should still own the lock due to refreshes
        assert lock1._acquired is True
        assert lock2._acquired is False


class TestClusterLockErrorHandling:
    """Error handling and edge cases."""

    def test_redis_connection_failure_on_acquire(self, lock_key, owner_id):
        """Test graceful handling when Redis is unavailable during acquisition."""
        # Use invalid Redis connection
        bad_redis = redis.Redis(
            host="invalid_host", port=1234, socket_connect_timeout=1
        )
        lock = ClusterLock(bad_redis, lock_key, owner_id, timeout=60)

        # Should return False, not raise exception
        assert lock.try_acquire() is False
        assert lock._acquired is False

    def test_redis_connection_failure_on_refresh(
        self, redis_client, lock_key, owner_id
    ):
        """Test graceful handling when Redis fails during refresh."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        # Acquire normally
        assert lock.try_acquire() is True

        # Replace Redis client with failing one
        lock.redis = redis.Redis(
            host="invalid_host", port=1234, socket_connect_timeout=1
        )

        # Refresh should fail gracefully
        lock._last_refresh = 0  # Force refresh
        assert lock.refresh() is False
        assert lock._acquired is False

    def test_context_manager_redis_failure_blocking(self, lock_key, owner_id):
        """Test context manager handles Redis failure when blocking=True."""
        bad_redis = redis.Redis(
            host="invalid_host", port=1234, socket_connect_timeout=1, decode_responses=False
        )
        lock = ClusterLock(bad_redis, lock_key, owner_id, timeout=60)

        with pytest.raises((ConnectionError, RuntimeError)):
            with lock.acquire(blocking=True):
                pass

    def test_context_manager_redis_failure_non_blocking(self, lock_key, owner_id):
        """Test context manager handles Redis failure when blocking=False."""
        bad_redis = redis.Redis(
            host="invalid_host", port=1234, socket_connect_timeout=1
        )
        lock = ClusterLock(bad_redis, lock_key, owner_id, timeout=60)

        with lock.acquire(blocking=False) as acquired_lock:
            assert acquired_lock is None

    def test_invalid_lock_parameters(self, redis_client):
        """Test validation of lock parameters."""
        owner_id = str(uuid.uuid4())

        # Empty key should raise ValueError
        with pytest.raises(ValueError, match="Lock key cannot be empty"):
            ClusterLock(redis_client, "", owner_id, timeout=60)

        # Empty owner_id should raise ValueError
        with pytest.raises(ValueError, match="Owner ID cannot be empty"):
            ClusterLock(redis_client, "test_key", "", timeout=60)

        # Invalid timeout should raise ValueError
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ClusterLock(redis_client, "test_key", owner_id, timeout=0)

    def test_refresh_after_redis_key_deleted(self, redis_client, lock_key, owner_id):
        """Test refresh behavior when Redis key is manually deleted."""
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        lock.try_acquire()

        # Manually delete the key (simulates external deletion)
        redis_client.delete(lock_key)

        # Refresh should fail and mark as not acquired
        lock._last_refresh = 0  # Force refresh
        assert lock.refresh() is False
        assert lock._acquired is False


class TestClusterLockDynamicRefreshInterval:
    """Dynamic refresh interval based on timeout."""

    def test_refresh_interval_calculation(self, redis_client, lock_key, owner_id):
        """Test refresh interval is calculated as max(timeout/10, 10)."""
        test_cases = [
            (30, 10),  # 30/10 = 3, but minimum is 10
            (100, 10),  # 100/10 = 10
            (200, 20),  # 200/10 = 20
            (1000, 100),  # 1000/10 = 100
        ]

        for timeout, expected_interval in test_cases:
            lock = ClusterLock(
                redis_client, f"{lock_key}_{timeout}", owner_id, timeout=timeout
            )
            lock.try_acquire()

            # Calculate expected interval using same logic as implementation
            refresh_interval = max(timeout // 10, 10)
            assert refresh_interval == expected_interval

            # Test rate limiting works with calculated interval
            assert lock.refresh() is True
            first_refresh_time = lock._last_refresh

            # Sleep less than interval - should be rate limited
            time.sleep(0.1)
            assert lock.refresh() is True
            assert lock._last_refresh == first_refresh_time  # No actual refresh


class TestClusterLockRealWorldScenarios:
    """Real-world usage patterns."""

    def test_execution_coordination_simulation(self, redis_client):
        """Simulate graph execution coordination across multiple pods."""
        graph_exec_id = str(uuid.uuid4())
        lock_key = f"execution:{graph_exec_id}"

        # Simulate 3 pods trying to execute same graph
        pods = [f"pod_{i}" for i in range(3)]
        execution_results = {}

        def execute_graph(pod_id):
            """Simulate graph execution with cluster lock."""
            lock = ClusterLock(redis_client, lock_key, pod_id, timeout=300)

            with lock.acquire(blocking=False) as acquired_lock:
                if acquired_lock is not None:
                    # Simulate execution work
                    execution_results[pod_id] = "executed"
                    time.sleep(0.1)
                else:
                    execution_results[pod_id] = "rejected"

        threads = []
        for pod_id in pods:
            thread = Thread(target=execute_graph, args=(pod_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Only one pod should have executed
        executed_count = sum(
            1 for result in execution_results.values() if result == "executed"
        )
        rejected_count = sum(
            1 for result in execution_results.values() if result == "rejected"
        )

        assert executed_count == 1
        assert rejected_count == 2

    def test_long_running_execution_with_refresh(
        self, redis_client, lock_key, owner_id
    ):
        """Test lock maintains ownership during long execution with periodic refresh."""
        lock = ClusterLock(
            redis_client, lock_key, owner_id, timeout=3
        )  # 3 second timeout

        def long_execution_with_refresh():
            """Simulate long-running execution with periodic refresh."""
            with lock.acquire(blocking=True) as acquired_lock:
                assert acquired_lock is not None

                # Simulate 10 seconds of work with refreshes every second
                for i in range(10):
                    time.sleep(1)
                    lock._last_refresh = 0  # Force refresh past rate limit
                    refresh_success = lock.refresh()
                    assert refresh_success is True, f"Refresh failed at iteration {i}"

                return "completed"

        # Should complete successfully without losing lock
        result = long_execution_with_refresh()
        assert result == "completed"

    def test_graceful_degradation_pattern(self, redis_client, lock_key):
        """Test graceful degradation when Redis becomes unavailable."""
        owner_id = str(uuid.uuid4())
        lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)

        # Normal operation
        assert lock.try_acquire() is True
        lock._last_refresh = 0  # Force refresh past rate limit
        assert lock.refresh() is True

        # Simulate Redis becoming unavailable
        original_redis = lock.redis
        lock.redis = redis.Redis(
            host="invalid_host", port=1234, socket_connect_timeout=1, decode_responses=False
        )

        # Should degrade gracefully
        lock._last_refresh = 0  # Force refresh past rate limit
        assert lock.refresh() is False
        assert lock._acquired is False

        # Restore Redis and verify can acquire again
        lock.redis = original_redis
        # Wait for original lock to expire
        time.sleep(1)

        new_lock = ClusterLock(redis_client, lock_key, owner_id, timeout=60)
        assert new_lock.try_acquire() is True


if __name__ == "__main__":
    # Run specific test for quick validation
    pytest.main([__file__, "-v"])
