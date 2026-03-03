"""Block management for agent generation.

Provides cached access to block metadata for validation and fixing.
"""

import logging
from typing import Any, Type

from backend.blocks import get_blocks as get_block_classes
from backend.blocks._base import Block

logger = logging.getLogger(__name__)

__all__ = ["get_blocks_as_dicts", "_reset_caches"]

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_blocks_cache: list[dict[str, Any]] | None = None


def _reset_caches() -> None:
    """Reset all module-level caches (useful for testing)."""
    global _blocks_cache
    _blocks_cache = None


# ---------------------------------------------------------------------------
# 1. get_blocks_as_dicts
# ---------------------------------------------------------------------------


def get_blocks_as_dicts() -> list[dict[str, Any]]:
    """Get all available blocks as dicts (cached after first call).

    Each dict contains the keys returned by ``Block.get_info().model_dump()``:
    id, name, description, inputSchema, outputSchema, categories,
    staticOutput, costs, contributors, uiType.

    Returns:
        List of block info dicts.
    """
    global _blocks_cache
    if _blocks_cache is not None:
        return _blocks_cache

    block_classes: dict[str, Type[Block]] = get_block_classes()  # type: ignore[assignment]
    blocks: list[dict[str, Any]] = []
    for block_cls in block_classes.values():
        try:
            instance = block_cls()
            info = instance.get_info().model_dump()
            blocks.append(info)
        except Exception:
            logger.warning(
                "Failed to load block info for %s, skipping",
                getattr(block_cls, "__name__", "unknown"),
                exc_info=True,
            )

    # Overlay LLM-optimized descriptions from the database (if available)
    try:
        from backend.util.clients import get_database_manager_client

        db_client = get_database_manager_client()
        optimized = db_client.get_optimized_block_descriptions()
        applied = 0
        for block in blocks:
            if opt_desc := optimized.get(block["id"]):
                block["description"] = opt_desc
                applied += 1
        if applied:
            logger.info("Applied %d optimized block descriptions", applied)
    except Exception:
        logger.debug(
            "Could not load optimized descriptions, using originals",
            exc_info=True,
        )

    _blocks_cache = blocks
    logger.info("Cached %d block dicts", len(blocks))
    return _blocks_cache
