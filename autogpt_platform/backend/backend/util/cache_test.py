"""
Unit tests for the TTL cache implementation.
"""

import threading
import time

from backend.util.cache import TTLCache, generate_cache_key


class TestTTLCache:
    """Tests for the TTLCache class."""

    def test_basic_get_set(self):
        """Test basic get and set operations."""
        cache = TTLCache(default_ttl=10)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test missing key
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = TTLCache(default_ttl=1)  # 1 second TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        cache = TTLCache(default_ttl=10)

        # Set with custom TTL
        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=5)

        # Both should be available immediately
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # After 1.1 seconds, key1 should expire
        time.sleep(1.1)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_max_size_eviction(self):
        """Test LRU eviction when max size is reached."""
        # Set a very small size limit (0.00025 MB = ~250 bytes)
        cache = TTLCache(default_ttl=10, max_size_mb=0.00025)

        # Add entries that will exceed the size limit
        # Each entry is about 71 bytes (55 + 16 buffer), so 3 entries = ~213 bytes
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.size() == 3

        # Adding another should trigger eviction (total would be ~284 bytes > 250 bytes)
        cache.set("key4", "value4")

        # Oldest entry (key1) should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_ordering(self):
        """Test that LRU ordering is maintained."""
        # Set a very small size limit to trigger eviction (0.00025 MB = ~250 bytes)
        cache = TTLCache(default_ttl=10, max_size_mb=0.00025)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to move it to end
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_clear(self):
        """Test clearing the cache."""
        cache = TTLCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = TTLCache(default_ttl=1, max_size_mb=5.0)

        cache.set("key1", "value1", ttl=10)
        cache.set("key2", "value2", ttl=1)  # Expires quickly

        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["max_size_mb"] == 5.0

        # Wait for key2 to expire
        time.sleep(1.1)
        stats = cache.stats()
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1

    def test_thread_safety(self):
        """Test thread-safe operations."""
        cache = TTLCache(default_ttl=10)
        errors = []

        def writer(start_idx):
            try:
                for i in range(100):
                    cache.set(f"key{start_idx + i}", f"value{start_idx + i}")
            except Exception as e:
                errors.append(e)

        def reader(start_idx):
            try:
                for i in range(100):
                    cache.get(f"key{start_idx + i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t1 = threading.Thread(target=writer, args=(i * 100,))
            t2 = threading.Thread(target=reader, args=(i * 100,))
            threads.extend([t1, t2])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_basic_key_generation(self):
        """Test basic cache key generation."""

        def sample_func(a, b, c=3):
            return a + b + c

        key1 = generate_cache_key(sample_func, (1, 2), {"c": 3})
        key2 = generate_cache_key(sample_func, (1, 2), {"c": 3})

        # Same inputs should generate same key
        assert key1 == key2

        # Different inputs should generate different keys
        key3 = generate_cache_key(sample_func, (1, 3), {"c": 3})
        assert key1 != key3

    def test_key_with_code_changes(self):
        """Test that code changes affect the cache key."""

        def func_v1():
            return 1

        def func_v2():
            return 2

        key1 = generate_cache_key(func_v1, (), {}, include_code=True)
        key2 = generate_cache_key(func_v2, (), {}, include_code=True)

        # Different function code should generate different keys
        assert key1 != key2

        # Without code, keys might be similar (same name)
        func_v2.__name__ = func_v1.__name__
        key3 = generate_cache_key(func_v1, (), {}, include_code=False)
        key4 = generate_cache_key(func_v2, (), {}, include_code=False)

        # Same function name without code check should generate same key
        assert key3 == key4

    def test_key_with_complex_args(self):
        """Test key generation with complex arguments."""

        def sample_func(data):
            return data

        # Test with dict
        key1 = generate_cache_key(sample_func, (), {"data": {"nested": {"value": 1}}})
        key2 = generate_cache_key(sample_func, (), {"data": {"nested": {"value": 1}}})
        assert key1 == key2

        # Test with list
        key3 = generate_cache_key(sample_func, ([1, 2, 3],), {})
        key4 = generate_cache_key(sample_func, ([1, 2, 3],), {})
        assert key3 == key4

        # Different order in list should generate different key
        key5 = generate_cache_key(sample_func, ([3, 2, 1],), {})
        assert key3 != key5

    def test_key_consistency(self):
        """Test that keys are consistent across calls."""

        def sample_func(a, b=2, c=3):
            return a + b + c

        # Different ways of passing the same arguments
        key1 = generate_cache_key(sample_func, (1,), {"b": 2, "c": 3})
        key2 = generate_cache_key(sample_func, (1,), {"c": 3, "b": 2})

        # Order of kwargs shouldn't matter (they're sorted)
        assert key1 == key2

    def test_non_serializable_objects(self):
        """Test key generation with non-JSON serializable objects."""

        def sample_func(obj):
            return str(obj)

        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomObject({self.value})"

        obj = CustomObject(42)

        # Should not raise an error
        key = generate_cache_key(sample_func, (obj,), {})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length
