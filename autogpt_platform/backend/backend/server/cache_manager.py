"""
Central cache management system for different components of the application.

This module provides separate cache instances for different parts of the system,
each with configurable size limits loaded from environment variables.
"""

import logging
from enum import Enum
from typing import Dict, Optional

from backend.util.cache import TTLCache
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class CacheComponent(str, Enum):
    """Enum of different cache components in the system."""

    STORE = "store"
    LIBRARY = "library"
    V1_API = "v1_api"
    BUILDER = "builder"
    OTTO = "otto"
    ADMIN = "admin"


class CacheManager:
    """
    Manages multiple cache instances for different components of the application.

    Each component gets its own cache instance with configurable size limits.
    """

    _instance: Optional["CacheManager"] = None
    _caches: Dict[CacheComponent, TTLCache] = {}

    def __new__(cls) -> "CacheManager":
        """Singleton pattern to ensure only one cache manager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize cache instances for each component."""
        settings = Settings()

        # Check if caching is enabled (default to True if not set)
        cache_enabled = getattr(settings, 'cache_enabled', True)
        if not cache_enabled:
            logger.info("[CACHE MANAGER] Caching is disabled globally")
            return

        # Create cache instances with configured sizes (in MB)
        # Use getattr with defaults in case settings don't have these attributes
        cache_configs = {
            CacheComponent.STORE: getattr(settings, 'cache_store_max_size_mb', 100.0),
            CacheComponent.LIBRARY: getattr(settings, 'cache_library_max_size_mb', 50.0),
            CacheComponent.V1_API: getattr(settings, 'cache_v1_api_max_size_mb', 200.0),
            CacheComponent.BUILDER: getattr(settings, 'cache_builder_max_size_mb', 50.0),
            CacheComponent.OTTO: getattr(settings, 'cache_otto_max_size_mb', 20.0),
            CacheComponent.ADMIN: getattr(settings, 'cache_admin_max_size_mb', 10.0),
        }

        default_ttl = getattr(settings, 'cache_default_ttl', 3600)
        for component, max_size_mb in cache_configs.items():
            self._caches[component] = TTLCache(
                default_ttl=default_ttl, max_size_mb=max_size_mb
            )
            logger.info(
                f"[CACHE MANAGER] Initialized {component.value} cache "
                f"(max_size={max_size_mb:.1f} MB, ttl={default_ttl}s)"
            )

    def get_cache(self, component: CacheComponent) -> Optional[TTLCache]:
        """
        Get the cache instance for a specific component.

        Args:
            component: The component to get cache for

        Returns:
            TTLCache instance or None if caching is disabled
        """
        settings = Settings()
        cache_enabled = getattr(settings, 'cache_enabled', True)
        if not cache_enabled:
            return None

        return self._caches.get(component)

    def invalidate_user_cache(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a specific user across all components.

        Args:
            user_id: The user ID to invalidate cache for

        Returns:
            Total number of keys invalidated
        """
        total_invalidated = 0
        for component, cache in self._caches.items():
            if cache:
                count = cache.invalidate_user(user_id)
                if count > 0:
                    logger.info(
                        f"[CACHE MANAGER] Invalidated {count} keys for user {user_id} "
                        f"in {component.value} cache"
                    )
                total_invalidated += count

        return total_invalidated

    def invalidate_pattern(
        self, pattern: str, component: Optional[CacheComponent] = None
    ) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Regular expression pattern to match
            component: Specific component to invalidate in, or None for all

        Returns:
            Total number of keys invalidated
        """
        total_invalidated = 0

        if component:
            cache = self._caches.get(component)
            if cache:
                total_invalidated = cache.invalidate_pattern(pattern)
        else:
            # Invalidate in all components
            for comp, cache in self._caches.items():
                if cache:
                    count = cache.invalidate_pattern(pattern)
                    if count > 0:
                        logger.info(
                            f"[CACHE MANAGER] Invalidated {count} keys matching '{pattern}' "
                            f"in {comp.value} cache"
                        )
                    total_invalidated += count

        return total_invalidated

    def invalidate_prefix(
        self, prefix: str, component: Optional[CacheComponent] = None
    ) -> int:
        """
        Invalidate cache entries with a specific prefix.

        Args:
            prefix: The prefix to match
            component: Specific component to invalidate in, or None for all

        Returns:
            Total number of keys invalidated
        """
        total_invalidated = 0

        if component:
            cache = self._caches.get(component)
            if cache:
                total_invalidated = cache.invalidate_prefix(prefix)
        else:
            # Invalidate in all components
            for comp, cache in self._caches.items():
                if cache:
                    count = cache.invalidate_prefix(prefix)
                    if count > 0:
                        logger.info(
                            f"[CACHE MANAGER] Invalidated {count} keys with prefix '{prefix}' "
                            f"in {comp.value} cache"
                        )
                    total_invalidated += count

        return total_invalidated

    def clear_all(self) -> None:
        """Clear all caches."""
        for component, cache in self._caches.items():
            if cache:
                cache.clear()
                logger.info(f"[CACHE MANAGER] Cleared {component.value} cache")

    def clear_component(self, component: CacheComponent) -> None:
        """
        Clear cache for a specific component.

        Args:
            component: The component to clear cache for
        """
        cache = self._caches.get(component)
        if cache:
            cache.clear()
            logger.info(f"[CACHE MANAGER] Cleared {component.value} cache")

    def get_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all cache components.

        Returns:
            Dictionary mapping component names to their statistics
        """
        stats = {}
        for component, cache in self._caches.items():
            if cache:
                stats[component.value] = cache.stats()
        return stats

    def get_component_stats(self, component: CacheComponent) -> Optional[Dict]:
        """
        Get statistics for a specific cache component.

        Args:
            component: The component to get stats for

        Returns:
            Cache statistics or None if component doesn't exist
        """
        cache = self._caches.get(component)
        return cache.stats() if cache else None


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_component_cache(component: CacheComponent) -> Optional[TTLCache]:
    """
    Get cache instance for a specific component.

    Args:
        component: The component to get cache for

    Returns:
        TTLCache instance or None if caching is disabled
    """
    return get_cache_manager().get_cache(component)
