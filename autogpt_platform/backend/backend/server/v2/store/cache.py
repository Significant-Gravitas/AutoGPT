import backend.server.v2.store.db
from backend.util.cache import cached

##############################################
############### Caches #######################
##############################################


def clear_all_caches():
    """Clear all caches."""
    _get_cached_store_agents.cache_clear()
    _get_cached_agent_details.cache_clear()
    _get_cached_store_creators.cache_clear()
    _get_cached_creator_details.cache_clear()


# Cache store agents list for 5 minutes
# Different cache entries for different query combinations
@cached(maxsize=5000, ttl_seconds=300, shared_cache=True)
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
@cached(maxsize=200, ttl_seconds=300, shared_cache=True)
async def _get_cached_agent_details(username: str, agent_name: str):
    """Cached helper to get agent details."""
    return await backend.server.v2.store.db.get_store_agent_details(
        username=username, agent_name=agent_name
    )


# Cache creators list for 5 minutes
@cached(maxsize=200, ttl_seconds=300, shared_cache=True)
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


# Cache individual creator details for 5 minutes
@cached(maxsize=100, ttl_seconds=300, shared_cache=True)
async def _get_cached_creator_details(username: str):
    """Cached helper to get creator details."""
    return await backend.server.v2.store.db.get_store_creator_details(
        username=username.lower()
    )
