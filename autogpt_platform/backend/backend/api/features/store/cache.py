import logging

from redis.exceptions import RedisClusterException, RedisError

from backend.api.features.search.embeddings import refused_public_queries
from backend.util.cache import cached

from . import db as store_db
from . import model as store_model

logger = logging.getLogger(__name__)

##############################################
############### Caches #######################
##############################################


def clear_all_caches():
    """Clear all caches."""
    _get_cached_store_agents.cache_clear()
    _get_cached_agent_details.cache_clear()
    _get_cached_store_creators.cache_clear()
    _get_cached_creator_details.cache_clear()


async def get_public_store_agents(
    featured: bool,
    creator: str | None,
    sorted_by: store_db.StoreAgentsSortOptions | None,
    search_query: str | None,
    category: str | None,
    page: int,
    page_size: int,
) -> store_model.StoreAgentsResponse:
    """Cached store agents for a public route, whose searches count against the store
    search embedding budget. A search ranked by text alone because that budget was
    spent is dropped from the cache, so it is ranked fully once the budget refills."""
    refused: list[str] = []
    token = refused_public_queries.set(refused)
    try:
        agents = await _get_cached_store_agents(
            featured=featured,
            creator=creator,
            sorted_by=sorted_by,
            search_query=search_query,
            category=category,
            page=page,
            page_size=page_size,
            public_search=True,
        )
    finally:
        refused_public_queries.reset(token)
    if refused:
        # The cache read and write already degrade on Redis trouble; this must too.
        try:
            _get_cached_store_agents.cache_delete(
                featured=featured,
                creator=creator,
                sorted_by=sorted_by,
                search_query=search_query,
                category=category,
                page=page,
                page_size=page_size,
                public_search=True,
            )
        except (RedisError, RedisClusterException, ConnectionError, OSError) as e:
            logger.warning(
                f"Could not drop a text-ranked store search from the cache: {e!r}"
            )
    return agents


# Cache store agents list for 5 minutes
# Different cache entries for different query combinations
@cached(maxsize=5000, ttl_seconds=300, shared_cache=True)
async def _get_cached_store_agents(
    featured: bool,
    creator: str | None,
    sorted_by: store_db.StoreAgentsSortOptions | None,
    search_query: str | None,
    category: str | None,
    page: int,
    page_size: int,
    public_search: bool = False,
):
    """Cached helper to get store agents."""
    return await store_db.get_store_agents(
        featured=featured,
        creators=[creator] if creator else None,
        sorted_by=sorted_by,
        search_query=search_query,
        category=category,
        page=page,
        page_size=page_size,
        public_search=public_search,
    )


# Cache individual agent details for 15 minutes
@cached(maxsize=200, ttl_seconds=300, shared_cache=True)
async def _get_cached_agent_details(
    username: str, agent_name: str, include_changelog: bool = False
):
    """Cached helper to get agent details."""
    return await store_db.get_store_agent_details(
        username=username, agent_name=agent_name, include_changelog=include_changelog
    )


# Cache creators list for 5 minutes
@cached(maxsize=200, ttl_seconds=300, shared_cache=True)
async def _get_cached_store_creators(
    featured: bool,
    search_query: str | None,
    sorted_by: store_db.StoreCreatorsSortOptions | None,
    page: int,
    page_size: int,
):
    """Cached helper to get store creators."""
    return await store_db.get_store_creators(
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
    return await store_db.get_store_creator(username=username.lower())
