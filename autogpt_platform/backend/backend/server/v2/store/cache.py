"""
Store-specific cache utilities.

This module provides cache invalidation helpers for store endpoints.
"""

import logging

from backend.server.cache_manager import CacheComponent, get_component_cache

logger = logging.getLogger(__name__)


def invalidate_user_profile_cache(user_id: str) -> None:
    """
    Invalidate store profile cache for a specific user.

    This should be called when a user's profile is updated
    to ensure they see fresh data immediately.

    Args:
        user_id: The user ID to invalidate cache for
    """
    store_cache = get_component_cache(CacheComponent.STORE)
    if store_cache:
        logger.info(
            f"[STORE CACHE] Attempting to invalidate profile cache for user {user_id}"
        )

        # Debug: Show what's in the cache
        cache_size = store_cache.size()
        logger.info(f"[STORE CACHE] Current cache size: {cache_size} entries")

        # Log first few cache keys for debugging
        if hasattr(store_cache, "_cache") and store_cache._cache:
            sample_keys = list(store_cache._cache.keys())[:3]
            for key in sample_keys:
                logger.info(f"[STORE CACHE] Sample cache key: {key}")

        # The cache key format is: module.function:user:user_id:hash
        # Example: backend.server.v2.store.routes.get_profile:user:7652f565-ef7a-40df-b5bf-d56c04d34f7f:005be36c29d3a4c9
        pattern = f".*get_profile.*{user_id}.*"

        count = store_cache.invalidate_pattern(pattern)

        if count > 0:
            logger.info(
                f"[STORE CACHE] Successfully invalidated {count} profile cache entries for user {user_id}"
            )
        else:
            # Try just the user_id pattern
            pattern = f".*{user_id}.*"
            count = store_cache.invalidate_pattern(pattern)
            if count > 0:
                logger.info(
                    f"[STORE CACHE] Successfully invalidated {count} cache entries for user {user_id} (broad match)"
                )
            else:
                logger.warning(
                    f"[STORE CACHE] No cache entries found to invalidate for user {user_id}"
                )
