"""
In-memory Redis client implementation as a fallback.
This allows running the platform without an external Redis server.
"""
import asyncio
import logging
from collections import defaultdict
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InMemoryRedis:
    """
    In-memory Redis-compatible client.
    Thread-safe singleton implementation.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data = {}
                    cls._instance._pubsub_channels = defaultdict(list)
                    cls._instance._data_lock = Lock()
        return cls._instance

    def ping(self) -> bool:
        """Test connection (always succeeds for in-memory)"""
        return True

    def get(self, key: str) -> Optional[str]:
        """Get a value by key"""
        with self._data_lock:
            return self._data.get(key)

    def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set a value with optional expiration"""
        with self._data_lock:
            if nx and key in self._data:
                return False
            if xx and key not in self._data:
                return False

            self._data[key] = str(value)
            # Note: Expiration (ex, px) not implemented in this simple version
            if ex or px:
                logger.debug(f"Expiration not implemented in InMemoryRedis (key: {key})")
            return True

    def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        with self._data_lock:
            count = 0
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    count += 1
            return count

    def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        with self._data_lock:
            return sum(1 for key in keys if key in self._data)

    def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching a pattern"""
        with self._data_lock:
            if pattern == "*":
                return list(self._data.keys())
            # Simple pattern matching
            import re
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            regex = re.compile(regex_pattern)
            return [key for key in self._data.keys() if regex.match(key)]

    def incr(self, key: str) -> int:
        """Increment a key"""
        with self._data_lock:
            current = int(self._data.get(key, 0))
            current += 1
            self._data[key] = str(current)
            return current

    def decr(self, key: str) -> int:
        """Decrement a key"""
        with self._data_lock:
            current = int(self._data.get(key, 0))
            current -= 1
            self._data[key] = str(current)
            return current

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key (not implemented)"""
        logger.debug(f"Expiration not implemented in InMemoryRedis (key: {key})")
        return key in self._data

    def ttl(self, key: str) -> int:
        """Get time to live (always returns -1 for no expiration)"""
        with self._data_lock:
            if key not in self._data:
                return -2  # Key doesn't exist
            return -1  # No expiration

    def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel"""
        # This is handled by InMemoryEventBus
        logger.debug(f"Publish to channel {channel}: {message[:100]}...")
        return 1

    def pubsub(self):
        """Get a pubsub object"""
        return InMemoryPubSub(self)

    def close(self):
        """Close connection (no-op for in-memory)"""
        pass

    def flushdb(self):
        """Flush all data (for testing)"""
        with self._data_lock:
            self._data.clear()


class InMemoryPubSub:
    """In-memory PubSub implementation"""

    def __init__(self, client: InMemoryRedis):
        self.client = client
        self.channels = []
        self.patterns = []

    def subscribe(self, *channels: str):
        """Subscribe to channels"""
        self.channels.extend(channels)
        logger.debug(f"Subscribed to channels: {channels}")

    def psubscribe(self, *patterns: str):
        """Subscribe to channel patterns"""
        self.patterns.extend(patterns)
        logger.debug(f"Subscribed to patterns: {patterns}")

    def unsubscribe(self, *channels: str):
        """Unsubscribe from channels"""
        for channel in channels:
            if channel in self.channels:
                self.channels.remove(channel)

    def punsubscribe(self, *patterns: str):
        """Unsubscribe from patterns"""
        for pattern in patterns:
            if pattern in self.patterns:
                self.patterns.remove(pattern)

    def listen(self):
        """Listen for messages (returns empty generator)"""
        # This should be integrated with InMemoryEventBus
        # For now, return empty to prevent blocking
        return iter([])

    def close(self):
        """Close pubsub connection"""
        self.channels = []
        self.patterns = []


class InMemoryAsyncRedis:
    """
    Async in-memory Redis-compatible client.
    Uses the same singleton storage as the sync version.
    """

    def __init__(self):
        self._sync_client = InMemoryRedis()

    async def ping(self) -> bool:
        """Test connection (always succeeds for in-memory)"""
        return True

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key"""
        return self._sync_client.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set a value with optional expiration"""
        return self._sync_client.set(key, value, ex, px, nx, xx)

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        return self._sync_client.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        return self._sync_client.exists(*keys)

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys matching a pattern"""
        return self._sync_client.keys(pattern)

    async def incr(self, key: str) -> int:
        """Increment a key"""
        return self._sync_client.incr(key)

    async def decr(self, key: str) -> int:
        """Decrement a key"""
        return self._sync_client.decr(key)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key (not implemented)"""
        return self._sync_client.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        """Get time to live"""
        return self._sync_client.ttl(key)

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel"""
        return self._sync_client.publish(channel, message)

    def pubsub(self):
        """Get a pubsub object"""
        return InMemoryAsyncPubSub(self)

    async def close(self):
        """Close connection (no-op for in-memory)"""
        pass

    async def flushdb(self):
        """Flush all data (for testing)"""
        self._sync_client.flushdb()


class InMemoryAsyncPubSub:
    """Async in-memory PubSub implementation"""

    def __init__(self, client: InMemoryAsyncRedis):
        self.client = client
        self.channels = []
        self.patterns = []

    async def subscribe(self, *channels: str):
        """Subscribe to channels"""
        self.channels.extend(channels)
        logger.debug(f"Subscribed to channels: {channels}")

    async def psubscribe(self, *patterns: str):
        """Subscribe to channel patterns"""
        self.patterns.extend(patterns)
        logger.debug(f"Subscribed to patterns: {patterns}")

    async def unsubscribe(self, *channels: str):
        """Unsubscribe from channels"""
        for channel in channels:
            if channel in self.channels:
                self.channels.remove(channel)

    async def punsubscribe(self, *patterns: str):
        """Unsubscribe from patterns"""
        for pattern in patterns:
            if pattern in self.patterns:
                self.patterns.remove(pattern)

    async def listen(self):
        """Listen for messages (returns empty async generator)"""
        # This should be integrated with InMemoryEventBus
        # For now, return empty to prevent blocking
        if False:
            yield

    async def close(self):
        """Close pubsub connection"""
        self.channels = []
        self.patterns = []
