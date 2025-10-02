"""
Cache functions for main V1 API endpoints.

This module contains all caching decorators and helpers for the V1 API,
separated from the main routes for better organization and maintainability.
"""

from typing import Sequence

from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data import user as user_db
from backend.data.block import get_blocks
from backend.util.cache import cached

# ===== Block Caches =====


# Cache block definitions with costs - they rarely change
@cached(maxsize=1, ttl_seconds=3600, shared_cache=True)
def get_cached_blocks() -> Sequence[dict]:
    """
    Get cached blocks with thundering herd protection.

    Uses cached decorator to prevent multiple concurrent requests
    from all executing the expensive block loading operation.
    """
    from backend.data.credit import get_block_cost

    block_classes = get_blocks()
    result = []

    for block_class in block_classes.values():
        block_instance = block_class()
        if not block_instance.disabled:
            # Get costs for this specific block class without creating another instance
            costs = get_block_cost(block_instance)
            result.append({**block_instance.to_dict(), "costs": costs})

    return result


# ===== Graph Caches =====


# Cache user's graphs list for 15 minutes
@cached(maxsize=1000, ttl_seconds=900, shared_cache=True)
async def get_cached_graphs(
    user_id: str,
    page: int,
    page_size: int,
):
    """Cached helper to get user's graphs."""
    return await graph_db.list_graphs_paginated(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# Cache individual graph details for 30 minutes
@cached(maxsize=500, ttl_seconds=1800, shared_cache=True)
async def get_cached_graph(
    graph_id: str,
    version: int | None,
    user_id: str,
):
    """Cached helper to get graph details."""
    return await graph_db.get_graph(
        graph_id=graph_id,
        version=version,
        user_id=user_id,
        include_subgraphs=True,  # needed to construct full credentials input schema
    )


# Cache graph versions for 30 minutes
@cached(maxsize=500, ttl_seconds=1800, shared_cache=True)
async def get_cached_graph_all_versions(
    graph_id: str,
    user_id: str,
) -> Sequence[graph_db.GraphModel]:
    """Cached helper to get all versions of a graph."""
    return await graph_db.get_graph_all_versions(
        graph_id=graph_id,
        user_id=user_id,
    )


# ===== Execution Caches =====


# Cache graph executions for 10 seconds.
@cached(maxsize=1000, ttl_seconds=10, shared_cache=True)
async def get_cached_graph_executions(
    graph_id: str,
    user_id: str,
    page: int,
    page_size: int,
):
    """Cached helper to get graph executions."""
    return await execution_db.get_graph_executions_paginated(
        graph_id=graph_id,
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# Cache all user executions for 10 seconds.
@cached(maxsize=500, ttl_seconds=10, shared_cache=True)
async def get_cached_graphs_executions(
    user_id: str,
    page: int,
    page_size: int,
):
    """Cached helper to get all user's graph executions."""
    return await execution_db.get_graph_executions_paginated(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


# Cache individual execution details for 10 seconds.
@cached(maxsize=1000, ttl_seconds=10, shared_cache=True)
async def get_cached_graph_execution(
    graph_exec_id: str,
    user_id: str,
):
    """Cached helper to get graph execution details."""
    return await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        include_node_executions=False,
    )


# ===== User Preference Caches =====


# Cache user timezone for 1 hour
@cached(maxsize=1000, ttl_seconds=3600, shared_cache=True)
async def get_cached_user_timezone(user_id: str):
    """Cached helper to get user timezone."""
    user = await user_db.get_user_by_id(user_id)
    return {"timezone": user.timezone if user else "UTC"}


# Cache user preferences for 30 minutes
@cached(maxsize=1000, ttl_seconds=1800, shared_cache=True)
async def get_cached_user_preferences(user_id: str):
    """Cached helper to get user notification preferences."""
    return await user_db.get_user_notification_preference(user_id)
