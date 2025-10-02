"""
Cache functions for Library API endpoints.

This module contains all caching decorators and helpers for the Library API,
separated from the main routes for better organization and maintainability.
"""

import backend.server.v2.library.db
from backend.util.cache import cached

# ===== Library Agent Caches =====


# Cache library agents list for 10 minutes
@cached(maxsize=1000, ttl_seconds=600, shared_cache=True)
async def get_cached_library_agents(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
):
    """Cached helper to get library agents list."""
    return await backend.server.v2.library.db.list_library_agents(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# Cache user's favorite agents for 5 minutes - favorites change more frequently
@cached(maxsize=500, ttl_seconds=300, shared_cache=True)
async def get_cached_library_agent_favorites(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
):
    """Cached helper to get user's favorite library agents."""
    return await backend.server.v2.library.db.list_favorite_library_agents(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# Cache individual library agent details for 30 minutes
@cached(maxsize=1000, ttl_seconds=1800, shared_cache=True)
async def get_cached_library_agent(
    library_agent_id: str,
    user_id: str,
):
    """Cached helper to get library agent details."""
    return await backend.server.v2.library.db.get_library_agent(
        id=library_agent_id,
        user_id=user_id,
    )


# Cache library agent by graph ID for 30 minutes
@cached(maxsize=1000, ttl_seconds=1800, shared_cache=True)
async def get_cached_library_agent_by_graph_id(
    graph_id: str,
    user_id: str,
):
    """Cached helper to get library agent by graph ID."""
    return await backend.server.v2.library.db.get_library_agent_by_graph_id(
        graph_id=graph_id,
        user_id=user_id,
    )


# Cache library agent by store version ID for 1 hour - marketplace agents are more stable
@cached(maxsize=500, ttl_seconds=3600, shared_cache=True)
async def get_cached_library_agent_by_store_version(
    store_listing_version_id: str,
    user_id: str,
):
    """Cached helper to get library agent by store version ID."""
    return await backend.server.v2.library.db.get_library_agent_by_store_version_id(
        store_listing_version_id=store_listing_version_id,
        user_id=user_id,
    )


# ===== Library Preset Caches =====


# Cache library presets list for 30 minutes
@cached(maxsize=500, ttl_seconds=1800, shared_cache=True)
async def get_cached_library_presets(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
):
    """Cached helper to get library presets list."""
    return await backend.server.v2.library.db.list_presets(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# Cache individual preset details for 30 minutes
@cached(maxsize=1000, ttl_seconds=1800, shared_cache=True)
async def get_cached_library_preset(
    preset_id: str,
    user_id: str,
):
    """Cached helper to get library preset details."""
    return await backend.server.v2.library.db.get_preset(
        preset_id=preset_id,
        user_id=user_id,
    )
