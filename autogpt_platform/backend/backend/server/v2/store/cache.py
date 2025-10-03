"""
Cache functions for Store API endpoints.

This module contains all caching decorators and helpers for the Store API,
separated from the main routes for better organization and maintainability.
"""

import backend.server.cache_config
import backend.server.v2.store.db
from backend.util.cache import cached


def _clear_submissions_cache(
    user_id: str, num_pages: int = backend.server.cache_config.MAX_PAGES_TO_CLEAR
):
    """
    Clear the submissions cache for the given user.

    Args:
        user_id: User ID whose cache should be cleared
        num_pages: Number of pages to clear (default from cache_config)
    """
    for page in range(1, num_pages + 1):
        _get_cached_submissions.cache_delete(
            user_id=user_id,
            page=page,
            page_size=backend.server.cache_config.V2_STORE_SUBMISSIONS_PAGE_SIZE,
        )


def _clear_my_agents_cache(
    user_id: str, num_pages: int = backend.server.cache_config.MAX_PAGES_TO_CLEAR
):
    """
    Clear the my agents cache for the given user.

    Args:
        user_id: User ID whose cache should be cleared
        num_pages: Number of pages to clear (default from cache_config)
    """
    for page in range(1, num_pages + 1):
        _get_cached_my_agents.cache_delete(
            user_id=user_id,
            page=page,
            page_size=backend.server.cache_config.V2_MY_AGENTS_PAGE_SIZE,
        )


# Cache user profiles for 1 hour per user
@cached(maxsize=1000, ttl_seconds=3600, shared_cache=True)
async def _get_cached_user_profile(user_id: str):
    """Cached helper to get user profile."""
    return await backend.server.v2.store.db.get_user_profile(user_id)


# Cache store agents list for 15 minutes
# Different cache entries for different query combinations
@cached(maxsize=5000, ttl_seconds=900, shared_cache=True)
async def _get_cached_store_agents(
    featured: bool,
    creator: str | None,
    sorted_by: str | None,
    search_query: str | None,
    category: str | None,
    page: int,
    page_size: int,
):
    """Cached helper to get store agents."""
    return await backend.server.v2.store.db.get_store_agents(
        featured=featured,
        creators=[creator] if creator else None,
        sorted_by=sorted_by,
        search_query=search_query,
        category=category,
        page=page,
        page_size=page_size,
    )


# Cache individual agent details for 15 minutes
@cached(maxsize=200, ttl_seconds=900, shared_cache=True)
async def _get_cached_agent_details(username: str, agent_name: str):
    """Cached helper to get agent details."""
    return await backend.server.v2.store.db.get_store_agent_details(
        username=username, agent_name=agent_name
    )


# Cache agent graphs for 1 hour
@cached(maxsize=200, ttl_seconds=3600, shared_cache=True)
async def _get_cached_agent_graph(store_listing_version_id: str):
    """Cached helper to get agent graph."""
    return await backend.server.v2.store.db.get_available_graph(
        store_listing_version_id
    )


# Cache agent by version for 1 hour
@cached(maxsize=200, ttl_seconds=3600, shared_cache=True)
async def _get_cached_store_agent_by_version(store_listing_version_id: str):
    """Cached helper to get store agent by version ID."""
    return await backend.server.v2.store.db.get_store_agent_by_version_id(
        store_listing_version_id
    )


# Cache creators list for 1 hour
@cached(maxsize=200, ttl_seconds=3600, shared_cache=True)
async def _get_cached_store_creators(
    featured: bool,
    search_query: str | None,
    sorted_by: str | None,
    page: int,
    page_size: int,
):
    """Cached helper to get store creators."""
    return await backend.server.v2.store.db.get_store_creators(
        featured=featured,
        search_query=search_query,
        sorted_by=sorted_by,
        page=page,
        page_size=page_size,
    )


# Cache individual creator details for 1 hour
@cached(maxsize=100, ttl_seconds=3600, shared_cache=True)
async def _get_cached_creator_details(username: str):
    """Cached helper to get creator details."""
    return await backend.server.v2.store.db.get_store_creator_details(
        username=username.lower()
    )


# Cache user's own agents for 5 mins (shorter TTL as this changes more frequently)
@cached(maxsize=500, ttl_seconds=300, shared_cache=True)
async def _get_cached_my_agents(user_id: str, page: int, page_size: int):
    """Cached helper to get user's agents."""
    return await backend.server.v2.store.db.get_my_agents(
        user_id, page=page, page_size=page_size
    )


# Cache user's submissions for 1 hour (shorter TTL as this changes frequently)
@cached(maxsize=500, ttl_seconds=3600, shared_cache=True)
async def _get_cached_submissions(user_id: str, page: int, page_size: int):
    """Cached helper to get user's submissions."""
    return await backend.server.v2.store.db.get_store_submissions(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )
