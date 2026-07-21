"""Block management for agent generation.

Provides cached access to block metadata for validation and fixing.
"""

import logging
from typing import Any, Type

from backend.blocks import get_blocks as get_block_classes
from backend.blocks._base import Block

logger = logging.getLogger(__name__)

__all__ = ["get_blocks_as_dicts", "reset_block_caches"]

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_blocks_cache: list[dict[str, Any]] | None = None


def reset_block_caches() -> None:
    """Reset all module-level caches (useful after updating block descriptions)."""
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
            # Use optimized description if available (loaded at startup)
            if instance.optimized_description:
                info["description"] = instance.optimized_description
            blocks.append(info)
        except Exception:
            logger.warning(
                "Failed to load block info for %s, skipping",
                getattr(block_cls, "__name__", "unknown"),
                exc_info=True,
            )

    _blocks_cache = blocks
    logger.info("Cached %d block dicts", len(blocks))
    return _blocks_cache
