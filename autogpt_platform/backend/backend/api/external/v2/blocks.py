"""
V2 External API - Blocks Endpoints

Provides read-only access to available building blocks.
"""

import logging

from fastapi import APIRouter, Response, Security
from fastapi.concurrency import run_in_threadpool
from prisma.enums import APIKeyPermission
from pydantic import BaseModel

from backend.api.external.middleware import require_permission
from backend.blocks import get_blocks
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.cache import cached
from backend.util.json import dumps

logger = logging.getLogger(__name__)

blocks_router = APIRouter()


# ============================================================================
# Internal Functions
# ============================================================================


def _compute_blocks_sync() -> str:
    """
    Synchronous function to compute blocks data.
    This does the heavy lifting: instantiate 226+ blocks, compute costs, serialize.
    """
    from backend.data.credit import get_block_cost

    block_classes = get_blocks()
    result = []

    for block_class in block_classes.values():
        block_instance = block_class()
        if not block_instance.disabled:
            costs = get_block_cost(block_instance)
            # Convert BlockCost BaseModel objects to dictionaries
            costs_dict = [
                cost.model_dump() if isinstance(cost, BaseModel) else cost
                for cost in costs
            ]
            result.append({**block_instance.to_dict(), "costs": costs_dict})

    return dumps(result)


@cached(ttl_seconds=3600)
async def _get_cached_blocks() -> str:
    """
    Async cached function with thundering herd protection.
    On cache miss: runs heavy work in thread pool
    On cache hit: returns cached string immediately
    """
    return await run_in_threadpool(_compute_blocks_sync)


# ============================================================================
# Endpoints
# ============================================================================


@blocks_router.get(
    path="",
    summary="List available blocks",
    responses={
        200: {
            "description": "List of available building blocks",
            "content": {
                "application/json": {
                    "schema": {
                        "items": {"additionalProperties": True, "type": "object"},
                        "type": "array",
                    }
                }
            },
        }
    },
)
async def list_blocks(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_BLOCK)
    ),
) -> Response:
    """
    List all available building blocks that can be used in graphs.

    Each block represents a specific capability (e.g., HTTP request, text processing,
    AI completion, etc.) that can be connected in a graph to create an agent.

    The response includes input/output schemas for each block, as well as
    cost information for blocks that consume credits.
    """
    content = await _get_cached_blocks()
    return Response(
        content=content,
        media_type="application/json",
    )
