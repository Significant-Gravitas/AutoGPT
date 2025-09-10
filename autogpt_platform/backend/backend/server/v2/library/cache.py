"""
Library-specific cache utilities.

This module provides cache invalidation helpers for library endpoints.
"""

import logging

from backend.server.cache_manager import CacheComponent, get_component_cache

logger = logging.getLogger(__name__)


def invalidate_user_library_cache(user_id: str) -> None:
    """
    Invalidate library cache for a specific user.

    This is called when a user adds/removes agents from their library
    to ensure they see fresh data.

    Args:
        user_id: The user ID to invalidate cache for
    """
    library_cache = get_component_cache(CacheComponent.LIBRARY)
    if library_cache:
        count = library_cache.invalidate_user(user_id)
        if count > 0:
            logger.info(
                f"[LIBRARY CACHE] Invalidated {count} cache entries for user {user_id}"
            )
