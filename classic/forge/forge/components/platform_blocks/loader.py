"""Local block loading from platform codebase.

This module provides functions to load and search platform blocks locally,
without making network calls. The platform backend is automatically discovered
from the monorepo structure.
"""

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_blocks_cache: dict[str, type] | None = None
_platform_path_added = False


def _get_platform_backend_path() -> Path | None:
    """Find the platform backend path relative to this file.

    The monorepo structure is:
        AutoGPT/main/
        ├── classic/forge/forge/components/platform_blocks/loader.py  (this file)
        └── autogpt_platform/backend/  (platform backend)
    """
    # This file is at: classic/forge/forge/components/platform_blocks/loader.py
    # Go up to classic/, then up to main/, then into autogpt_platform/backend/
    this_file = Path(__file__).resolve()
    classic_dir = this_file.parent.parent.parent.parent.parent.parent  # classic/
    main_dir = classic_dir.parent  # main/
    platform_backend = main_dir / "autogpt_platform" / "backend"

    if platform_backend.exists() and (platform_backend / "backend").exists():
        return platform_backend
    return None


def _ensure_platform_path() -> bool:
    """Add platform backend to sys.path if not already present."""
    global _platform_path_added
    if _platform_path_added:
        return True

    platform_path = _get_platform_backend_path()
    if platform_path is None:
        logger.debug("Platform backend not found in monorepo structure")
        return False

    path_str = str(platform_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.debug(f"Added platform backend to path: {path_str}")

    _platform_path_added = True
    return True


def is_platform_available() -> bool:
    """Check if platform blocks can be imported."""
    _ensure_platform_path()
    try:
        from backend.blocks import (  # pyright: ignore[reportMissingImports]
            load_all_blocks,
        )

        _ = load_all_blocks  # Silence unused import warning
        return True
    except ImportError:
        return False


def load_blocks() -> dict[str, type]:
    """Load all blocks from platform codebase.

    Returns:
        Dictionary mapping block IDs to block classes.
    """
    global _blocks_cache
    if _blocks_cache is not None:
        return _blocks_cache

    _ensure_platform_path()

    try:
        from backend.blocks import (  # pyright: ignore[reportMissingImports]
            load_all_blocks,
        )

        loaded: dict[str, type] = load_all_blocks()
        _blocks_cache = loaded
        logger.info(f"Loaded {len(loaded)} platform blocks")
        return loaded
    except ImportError as e:
        logger.warning(f"Could not import platform blocks: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading platform blocks: {e}")
        return {}


def get_block(block_id: str) -> Any | None:
    """Get a specific block instance by ID.

    Args:
        block_id: The unique block ID (UUID format).

    Returns:
        Block instance or None if not found.
    """
    blocks = load_blocks()
    block_cls = blocks.get(block_id)
    if block_cls:
        return block_cls()
    return None


def search_blocks(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search blocks by name or description.

    Args:
        query: Search query (case-insensitive).
        limit: Maximum number of results to return.

    Returns:
        List of block info dictionaries.
    """
    blocks = load_blocks()
    results: list[dict[str, Any]] = []
    query_lower = query.lower()

    for block_id, block_cls in blocks.items():
        try:
            block = block_cls()
        except Exception as e:
            logger.warning(f"Could not instantiate block {block_id}: {e}")
            continue

        # Skip disabled blocks
        if getattr(block, "disabled", False):
            continue

        # Get name and description
        name = getattr(block, "name", block_cls.__name__)
        description = getattr(block, "description", "")

        # Check name and description for match
        name_match = query_lower in name.lower()
        desc_match = query_lower in description.lower()

        # Check categories
        categories = []
        if hasattr(block, "categories"):
            categories = [c.value for c in block.categories]
        category_match = any(query_lower in c.lower() for c in categories)

        if name_match or desc_match or category_match:
            # Get input schema
            input_schema: dict[str, Any] = {}
            if hasattr(block, "input_schema"):
                try:
                    input_schema = block.input_schema.jsonschema()
                except Exception:
                    pass

            # Get output schema
            output_schema: dict[str, Any] = {}
            if hasattr(block, "output_schema"):
                try:
                    output_schema = block.output_schema.jsonschema()
                except Exception:
                    pass

            results.append(
                {
                    "id": block_id,
                    "name": name,
                    "description": description,
                    "categories": categories,
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                }
            )

            if len(results) >= limit:
                break

    return results


def get_block_info(block_id: str) -> dict[str, Any] | None:
    """Get detailed information about a specific block.

    Args:
        block_id: The unique block ID (UUID format).

    Returns:
        Block info dictionary or None if not found.
    """
    block = get_block(block_id)
    if not block:
        return None

    name = getattr(block, "name", block.__class__.__name__)
    description = getattr(block, "description", "")

    categories = []
    if hasattr(block, "categories"):
        categories = [c.value for c in block.categories]

    input_schema: dict[str, Any] = {}
    if hasattr(block, "input_schema"):
        try:
            input_schema = block.input_schema.jsonschema()
        except Exception:
            pass

    output_schema: dict[str, Any] = {}
    if hasattr(block, "output_schema"):
        try:
            output_schema = block.output_schema.jsonschema()
        except Exception:
            pass

    return {
        "id": block_id,
        "name": name,
        "description": description,
        "categories": categories,
        "input_schema": input_schema,
        "output_schema": output_schema,
    }


def clear_cache() -> None:
    """Clear the blocks cache. Useful for testing."""
    global _blocks_cache, _platform_path_added
    _blocks_cache = None
    _platform_path_added = False
