"""
Store listing embedding helpers.

Thin store-specific wrappers around the generic embedding service in
``backend.api.features.search.embeddings``. These exist for backward
compatibility with code that addresses store listings by
``StoreListingVersion.id`` directly.
"""

import logging
from typing import Any

import prisma
from prisma.enums import ContentType

from backend.api.features.search.embeddings import (
    backfill_all_content_types,
    build_searchable_text,
    delete_content_embedding,
    generate_embedding,
    get_content_embedding,
    store_content_embedding,
)

logger = logging.getLogger(__name__)


async def store_embedding(
    version_id: str,
    embedding: list[float],
    tx: prisma.Prisma | None = None,
) -> bool:
    """
    Store embedding in the database.

    BACKWARD COMPATIBILITY: Maintained for existing store listing usage.
    DEPRECATED: Use ensure_embedding() instead (includes searchable_text).
    """
    return await store_content_embedding(
        content_type=ContentType.STORE_AGENT,
        content_id=version_id,
        embedding=embedding,
        searchable_text="",  # Empty for backward compat; ensure_embedding() populates this
        metadata=None,
        user_id=None,  # Store agents are public
        tx=tx,
    )


async def get_embedding(version_id: str) -> dict[str, Any] | None:
    """
    Retrieve embedding record for a listing version.

    BACKWARD COMPATIBILITY: Maintained for existing store listing usage.
    Returns dict with storeListingVersionId, embedding, timestamps or None if not found.
    """
    result = await get_content_embedding(
        ContentType.STORE_AGENT, version_id, user_id=None
    )
    if result:
        # Transform to old format for backward compatibility
        return {
            "storeListingVersionId": result["contentId"],
            "embedding": result["embedding"],
            "createdAt": result["createdAt"],
            "updatedAt": result["updatedAt"],
        }
    return None


async def ensure_embedding(
    version_id: str,
    name: str,
    description: str,
    sub_heading: str,
    categories: list[str],
    force: bool = False,
    tx: prisma.Prisma | None = None,
) -> bool:
    """
    Ensure an embedding exists for the listing version.

    Creates embedding if missing. Use force=True to regenerate.
    Backward-compatible wrapper for store listings.

    Args:
        version_id: The StoreListingVersion ID
        name: Agent name
        description: Agent description
        sub_heading: Agent sub-heading
        categories: Agent categories
        force: Force regeneration even if embedding exists
        tx: Optional transaction client

    Returns:
        True if embedding exists/was created

    Raises exceptions on failure - caller should handle.
    """
    # Check if embedding already exists
    if not force:
        existing = await get_embedding(version_id)
        if existing and existing.get("embedding"):
            logger.debug(f"Embedding for version {version_id} already exists")
            return True

    # Build searchable text for embedding
    searchable_text = build_searchable_text(name, description, sub_heading, categories)

    # Generate new embedding
    embedding = await generate_embedding(searchable_text)

    # Store the embedding with metadata using new function
    metadata = {
        "name": name,
        "subHeading": sub_heading,
        "categories": categories,
    }
    return await store_content_embedding(
        content_type=ContentType.STORE_AGENT,
        content_id=version_id,
        embedding=embedding,
        searchable_text=searchable_text,
        metadata=metadata,
        user_id=None,  # Store agents are public
        tx=tx,
    )


async def delete_embedding(version_id: str) -> bool:
    """
    Delete embedding for a listing version.

    BACKWARD COMPATIBILITY: Maintained for existing store listing usage.
    Note: This is usually handled automatically by CASCADE delete,
    but provided for manual cleanup if needed.
    """
    return await delete_content_embedding(ContentType.STORE_AGENT, version_id)


async def backfill_missing_embeddings(batch_size: int = 10) -> dict[str, Any]:
    """
    Generate embeddings for approved listings that don't have them.

    BACKWARD COMPATIBILITY: Maintained for existing usage.
    This now delegates to backfill_all_content_types() to process all content types.

    Args:
        batch_size: Number of embeddings to generate per content type

    Returns:
        Dict with success/failure counts aggregated across all content types
    """
    # Delegate to the new generic backfill system
    result = await backfill_all_content_types(batch_size)

    # Return in the old format for backward compatibility
    return result["totals"]
