"""
V2 External API - Blocks Endpoints

Provides read-only access to available building blocks.
"""

import logging

from fastapi import APIRouter, Security
from fastapi.concurrency import run_in_threadpool
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.blocks import get_blocks
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.cache import cached

from .models import BlockInfo

logger = logging.getLogger(__name__)

blocks_router = APIRouter()


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
)
async def list_blocks(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_BLOCK)
    ),
) -> list[BlockInfo]:
    """
    List all available building blocks that can be used in graphs.

    Each block represents a specific capability (e.g., HTTP request, text processing,
    AI completion, etc.) that can be connected in a graph to create an agent.

    The response includes input/output schemas for each block, as well as
    cost information for blocks that consume credits.
    """
    return await _get_cached_blocks()
