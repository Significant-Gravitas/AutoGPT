"""
V2 External API - Blocks Endpoints

Provides read-only access to available building blocks.
"""

import logging

from fastapi import APIRouter, Depends, Security
from fastapi.concurrency import run_in_threadpool
from prisma.enums import APIKeyPermission

from backend.blocks import get_blocks
from backend.util.cache import cached

from .models import BlockInfo
from .pagination import Page, PageRequest, page_request
from .tenancy import TenantContext, require_permission

logger = logging.getLogger(__name__)

blocks_router = APIRouter(tags=["blocks"])


# ============================================================================
# Internal Functions
# ============================================================================


def _compute_blocks_sync() -> list[BlockInfo]:
    """
    Synchronous function to compute blocks data.
    This does the heavy lifting: instantiate 226+ blocks, compute costs, serialize.
    """
    return [
        BlockInfo.from_internal(block)
        for block_class in get_blocks().values()
        if not (block := block_class()).disabled
    ]


@cached(ttl_seconds=3600)
async def _get_cached_blocks() -> list[BlockInfo]:
    """
    Async cached function with thundering herd protection.
    On cache miss: runs heavy work in thread pool
    On cache hit: returns cached list immediately
    """
    return await run_in_threadpool(_compute_blocks_sync)


# ============================================================================
# Endpoints
# ============================================================================


@blocks_router.get(
    path="",
    summary="List available blocks",
    operation_id="listAvailableBlocks",
)
async def list_available_blocks(
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_BLOCK)),
) -> Page[BlockInfo]:
    """List all available blocks with their input/output schemas and cost information."""
    return page.slice(await _get_cached_blocks())
