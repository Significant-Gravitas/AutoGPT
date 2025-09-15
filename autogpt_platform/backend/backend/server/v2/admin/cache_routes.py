"""
Admin endpoints for managing the cache system.

This module provides administrative endpoints for monitoring and managing
the application's cache system.
"""

import logging
from typing import Dict, Optional

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, HTTPException, Path, Query, Security, status
from pydantic import BaseModel

from backend.server.cache_manager import CacheComponent, get_cache_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cache",
    tags=["admin", "cache"],
    dependencies=[Security(requires_admin_user)],
)


class CacheStats(BaseModel):
    """Cache statistics response model."""

    component: str
    total_entries: int
    valid_entries: int
    expired_entries: int
    max_size_mb: Optional[float]
    current_size_mb: float


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    overall: Dict[str, CacheStats]
    total_entries_all: int
    total_valid_all: int


class CacheInvalidationResponse(BaseModel):
    """Response model for cache invalidation operations."""

    keys_invalidated: int
    message: str


@router.get(
    "/stats",
    summary="Get cache statistics",
    response_model=CacheStatsResponse,
)
async def get_cache_stats() -> CacheStatsResponse:
    """
    Get statistics for all cache components.

    Returns:
        CacheStatsResponse: Statistics for each cache component
    """
    cache_manager = get_cache_manager()
    all_stats = cache_manager.get_stats()

    overall_stats = {}
    total_entries = 0
    total_valid = 0

    for component, stats in all_stats.items():
        overall_stats[component] = CacheStats(
            component=component,
            total_entries=stats["total_entries"],
            valid_entries=stats["valid_entries"],
            expired_entries=stats["expired_entries"],
            max_size_mb=stats.get("max_size_mb"),
            current_size_mb=stats.get("current_size_mb", 0.0),
        )
        total_entries += stats["total_entries"]
        total_valid += stats["valid_entries"]

    return CacheStatsResponse(
        overall=overall_stats,
        total_entries_all=total_entries,
        total_valid_all=total_valid,
    )


@router.get(
    "/stats/{component}",
    summary="Get cache statistics for a specific component",
    response_model=CacheStats,
)
async def get_component_cache_stats(
    component: CacheComponent = Path(
        ..., description="Cache component to get stats for"
    )
) -> CacheStats:
    """
    Get statistics for a specific cache component.

    Args:
        component: The cache component to get statistics for

    Returns:
        CacheStats: Statistics for the specified component

    Raises:
        HTTPException(404): If the component doesn't exist
    """
    cache_manager = get_cache_manager()
    stats = cache_manager.get_component_stats(component)

    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache component '{component}' not found",
        )

    return CacheStats(
        component=component.value,
        total_entries=stats["total_entries"],
        valid_entries=stats["valid_entries"],
        expired_entries=stats["expired_entries"],
        max_size_mb=stats.get("max_size_mb"),
        current_size_mb=stats.get("current_size_mb", 0.0),
    )


@router.post(
    "/clear",
    summary="Clear all caches",
    response_model=CacheInvalidationResponse,
)
async def clear_all_caches() -> CacheInvalidationResponse:
    """
    Clear all cache entries across all components.

    Returns:
        CacheInvalidationResponse: Result of the clear operation
    """
    cache_manager = get_cache_manager()

    # Get stats before clearing
    stats_before = cache_manager.get_stats()
    total_before = sum(s["total_entries"] for s in stats_before.values())

    # Clear all caches
    cache_manager.clear_all()

    logger.info(f"[ADMIN] Cleared all caches, removed {total_before} entries")

    return CacheInvalidationResponse(
        keys_invalidated=total_before,
        message=f"Successfully cleared all caches, removed {total_before} entries",
    )


@router.post(
    "/clear/{component}",
    summary="Clear cache for a specific component",
    response_model=CacheInvalidationResponse,
)
async def clear_component_cache(
    component: CacheComponent = Path(..., description="Cache component to clear")
) -> CacheInvalidationResponse:
    """
    Clear all cache entries for a specific component.

    Args:
        component: The cache component to clear

    Returns:
        CacheInvalidationResponse: Result of the clear operation

    Raises:
        HTTPException(404): If the component doesn't exist
    """
    cache_manager = get_cache_manager()

    # Get stats before clearing
    stats_before = cache_manager.get_component_stats(component)
    if stats_before is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache component '{component}' not found",
        )

    entries_before = stats_before["total_entries"]

    # Clear the component cache
    cache_manager.clear_component(component)

    logger.info(
        f"[ADMIN] Cleared {component.value} cache, removed {entries_before} entries"
    )

    return CacheInvalidationResponse(
        keys_invalidated=entries_before,
        message=f"Successfully cleared {component.value} cache, removed {entries_before} entries",
    )


@router.post(
    "/invalidate/user/{user_id}",
    summary="Invalidate all cache for a specific user",
    response_model=CacheInvalidationResponse,
)
async def invalidate_user_cache(
    user_id: str = Path(..., description="User ID to invalidate cache for")
) -> CacheInvalidationResponse:
    """
    Invalidate all cache entries for a specific user across all components.

    Args:
        user_id: The user ID to invalidate cache for

    Returns:
        CacheInvalidationResponse: Result of the invalidation operation
    """
    cache_manager = get_cache_manager()
    keys_invalidated = cache_manager.invalidate_user_cache(user_id)

    logger.info(
        f"[ADMIN] Invalidated {keys_invalidated} cache entries for user {user_id}"
    )

    return CacheInvalidationResponse(
        keys_invalidated=keys_invalidated,
        message=f"Successfully invalidated {keys_invalidated} cache entries for user {user_id}",
    )


@router.post(
    "/invalidate/pattern",
    summary="Invalidate cache entries matching a pattern",
    response_model=CacheInvalidationResponse,
)
async def invalidate_pattern(
    pattern: str = Query(..., description="Regular expression pattern to match"),
    component: Optional[CacheComponent] = Query(
        None, description="Specific component to invalidate in (all if not specified)"
    ),
) -> CacheInvalidationResponse:
    """
    Invalidate cache entries matching a regular expression pattern.

    Args:
        pattern: Regular expression pattern to match cache keys
        component: Optional specific component to invalidate in

    Returns:
        CacheInvalidationResponse: Result of the invalidation operation
    """
    cache_manager = get_cache_manager()
    keys_invalidated = cache_manager.invalidate_pattern(pattern, component)

    component_msg = f" in {component.value}" if component else " across all components"
    logger.info(
        f"[ADMIN] Invalidated {keys_invalidated} cache entries matching '{pattern}'{component_msg}"
    )

    return CacheInvalidationResponse(
        keys_invalidated=keys_invalidated,
        message=f"Successfully invalidated {keys_invalidated} cache entries matching pattern '{pattern}'{component_msg}",
    )


@router.post(
    "/invalidate/prefix",
    summary="Invalidate cache entries with a specific prefix",
    response_model=CacheInvalidationResponse,
)
async def invalidate_prefix(
    prefix: str = Query(..., description="Prefix to match"),
    component: Optional[CacheComponent] = Query(
        None, description="Specific component to invalidate in (all if not specified)"
    ),
) -> CacheInvalidationResponse:
    """
    Invalidate cache entries with a specific prefix.

    Args:
        prefix: Prefix to match cache keys
        component: Optional specific component to invalidate in

    Returns:
        CacheInvalidationResponse: Result of the invalidation operation
    """
    cache_manager = get_cache_manager()
    keys_invalidated = cache_manager.invalidate_prefix(prefix, component)

    component_msg = f" in {component.value}" if component else " across all components"
    logger.info(
        f"[ADMIN] Invalidated {keys_invalidated} cache entries with prefix '{prefix}'{component_msg}"
    )

    return CacheInvalidationResponse(
        keys_invalidated=keys_invalidated,
        message=f"Successfully invalidated {keys_invalidated} cache entries with prefix '{prefix}'{component_msg}",
    )
