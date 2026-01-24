"""
Unified Content Embeddings Service

Handles generation and storage of OpenAI embeddings for all content types
(store listings, blocks, documentation, library agents) to enable semantic/hybrid search.
"""

import asyncio
import logging
import time
from typing import Any

import prisma
from prisma.enums import ContentType
from tiktoken import encoding_for_model

from backend.api.features.store.content_handlers import CONTENT_HANDLERS
from backend.data.db import execute_raw_with_schema, query_raw_with_schema
from backend.util.clients import get_openai_client
from backend.util.json import dumps

logger = logging.getLogger(__name__)

# OpenAI embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
# Embedding dimension for the model above
# text-embedding-3-small: 1536, text-embedding-3-large: 3072
EMBEDDING_DIM = 1536
# OpenAI embedding token limit (8,191 with 1 token buffer for safety)
EMBEDDING_MAX_TOKENS = 8191


def build_searchable_text(
    name: str,
    description: str,
    sub_heading: str,
    categories: list[str],
) -> str:
    """
    Build searchable text from listing version fields.

    Combines relevant fields into a single string for embedding.
    """
    parts = []

    # Name is important - include it
    if name:
        parts.append(name)

    # Sub-heading provides context
    if sub_heading:
        parts.append(sub_heading)

    # Description is the main content
    if description:
        parts.append(description)

    # Categories help with semantic matching
    if categories:
        parts.append(" ".join(categories))

    return " ".join(parts)


async def generate_embedding(text: str) -> list[float]:
    """
    Generate embedding for text using OpenAI API.

    Raises exceptions on failure - caller should handle.
    """
    client = get_openai_client()
    if not client:
        raise RuntimeError("openai_internal_api_key not set, cannot generate embedding")

    # Truncate text to token limit using tiktoken
    # Character-based truncation is insufficient because token ratios vary by content type
    enc = encoding_for_model(EMBEDDING_MODEL)
    tokens = enc.encode(text)
    if len(tokens) > EMBEDDING_MAX_TOKENS:
        tokens = tokens[:EMBEDDING_MAX_TOKENS]
        truncated_text = enc.decode(tokens)
        logger.info(
            f"Truncated text from {len(enc.encode(text))} to {len(tokens)} tokens"
        )
    else:
        truncated_text = text

    start_time = time.time()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated_text,
    )
    latency_ms = (time.time() - start_time) * 1000

    embedding = response.data[0].embedding
    logger.info(
        f"Generated embedding: {len(embedding)} dims, "
        f"{len(tokens)} tokens, {latency_ms:.0f}ms"
    )
    return embedding


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


async def store_content_embedding(
    content_type: ContentType,
    content_id: str,
    embedding: list[float],
    searchable_text: str,
    metadata: dict | None = None,
    user_id: str | None = None,
    tx: prisma.Prisma | None = None,
) -> bool:
    """
    Store embedding in the unified content embeddings table.

    New function for unified content embedding storage.
    Uses raw SQL since Prisma doesn't natively support pgvector.

    Raises exceptions on failure - caller should handle.
    """
    client = tx if tx else prisma.get_client()

    # Convert embedding to PostgreSQL vector format
    embedding_str = embedding_to_vector_string(embedding)
    metadata_json = dumps(metadata or {})

    # Upsert the embedding
    # WHERE clause in DO UPDATE prevents PostgreSQL 15 bug with NULLS NOT DISTINCT
    # Use unqualified ::vector - pgvector is in search_path on all environments
    await execute_raw_with_schema(
        """
        INSERT INTO {schema_prefix}"UnifiedContentEmbedding" (
            "id", "contentType", "contentId", "userId", "embedding", "searchableText", "metadata", "createdAt", "updatedAt"
        )
        VALUES (gen_random_uuid()::text, $1::{schema_prefix}"ContentType", $2, $3, $4::vector, $5, $6::jsonb, NOW(), NOW())
        ON CONFLICT ("contentType", "contentId", "userId")
        DO UPDATE SET
            "embedding" = $4::vector,
            "searchableText" = $5,
            "metadata" = $6::jsonb,
            "updatedAt" = NOW()
        WHERE {schema_prefix}"UnifiedContentEmbedding"."contentType" = $1::{schema_prefix}"ContentType"
            AND {schema_prefix}"UnifiedContentEmbedding"."contentId" = $2
            AND ({schema_prefix}"UnifiedContentEmbedding"."userId" = $3 OR ($3 IS NULL AND {schema_prefix}"UnifiedContentEmbedding"."userId" IS NULL))
        """,
        content_type,
        content_id,
        user_id,
        embedding_str,
        searchable_text,
        metadata_json,
        client=client,
    )

    logger.info(f"Stored embedding for {content_type}:{content_id}")
    return True


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


async def get_content_embedding(
    content_type: ContentType, content_id: str, user_id: str | None = None
) -> dict[str, Any] | None:
    """
    Retrieve embedding record for any content type.

    New function for unified content embedding retrieval.
    Returns dict with contentType, contentId, embedding, timestamps or None if not found.

    Raises exceptions on failure - caller should handle.
    """
    result = await query_raw_with_schema(
        """
        SELECT
            "contentType",
            "contentId",
            "userId",
            "embedding"::text as "embedding",
            "searchableText",
            "metadata",
            "createdAt",
            "updatedAt"
        FROM {schema_prefix}"UnifiedContentEmbedding"
        WHERE "contentType" = $1::{schema_prefix}"ContentType" AND "contentId" = $2 AND ("userId" = $3 OR ($3 IS NULL AND "userId" IS NULL))
        """,
        content_type,
        content_id,
        user_id,
    )

    if result and len(result) > 0:
        return result[0]
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


async def delete_content_embedding(
    content_type: ContentType, content_id: str, user_id: str | None = None
) -> bool:
    """
    Delete embedding for any content type.

    New function for unified content embedding deletion.
    Note: This is usually handled automatically by CASCADE delete,
    but provided for manual cleanup if needed.

    Args:
        content_type: The type of content (STORE_AGENT, LIBRARY_AGENT, etc.)
        content_id: The unique identifier for the content
        user_id: Optional user ID. For public content (STORE_AGENT, BLOCK), pass None.
                 For user-scoped content (LIBRARY_AGENT), pass the user's ID to avoid
                 deleting embeddings belonging to other users.

    Returns:
        True if deletion succeeded, False otherwise
    """
    try:
        client = prisma.get_client()

        await execute_raw_with_schema(
            """
            DELETE FROM {schema_prefix}"UnifiedContentEmbedding"
            WHERE "contentType" = $1::{schema_prefix}"ContentType"
              AND "contentId" = $2
              AND ("userId" = $3 OR ($3 IS NULL AND "userId" IS NULL))
            """,
            content_type,
            content_id,
            user_id,
            client=client,
        )

        user_str = f" (user: {user_id})" if user_id else ""
        logger.info(f"Deleted embedding for {content_type}:{content_id}{user_str}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete embedding for {content_type}:{content_id}: {e}")
        return False


async def get_embedding_stats() -> dict[str, Any]:
    """
    Get statistics about embedding coverage for all content types.

    Returns stats per content type and overall totals.
    """
    try:
        stats_by_type = {}
        total_items = 0
        total_with_embeddings = 0
        total_without_embeddings = 0

        # Aggregate stats from all handlers
        for content_type, handler in CONTENT_HANDLERS.items():
            try:
                stats = await handler.get_stats()
                stats_by_type[content_type.value] = {
                    "total": stats["total"],
                    "with_embeddings": stats["with_embeddings"],
                    "without_embeddings": stats["without_embeddings"],
                    "coverage_percent": (
                        round(stats["with_embeddings"] / stats["total"] * 100, 1)
                        if stats["total"] > 0
                        else 0
                    ),
                }

                total_items += stats["total"]
                total_with_embeddings += stats["with_embeddings"]
                total_without_embeddings += stats["without_embeddings"]

            except Exception as e:
                logger.error(f"Failed to get stats for {content_type.value}: {e}")
                stats_by_type[content_type.value] = {
                    "total": 0,
                    "with_embeddings": 0,
                    "without_embeddings": 0,
                    "coverage_percent": 0,
                    "error": str(e),
                }

        return {
            "by_type": stats_by_type,
            "totals": {
                "total": total_items,
                "with_embeddings": total_with_embeddings,
                "without_embeddings": total_without_embeddings,
                "coverage_percent": (
                    round(total_with_embeddings / total_items * 100, 1)
                    if total_items > 0
                    else 0
                ),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        return {
            "by_type": {},
            "totals": {
                "total": 0,
                "with_embeddings": 0,
                "without_embeddings": 0,
                "coverage_percent": 0,
            },
            "error": str(e),
        }


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


async def backfill_all_content_types(batch_size: int = 10) -> dict[str, Any]:
    """
    Generate embeddings for all content types using registered handlers.

    Processes content types in order: BLOCK → STORE_AGENT → DOCUMENTATION.
    This ensures foundational content (blocks) are searchable first.

    Args:
        batch_size: Number of embeddings to generate per content type

    Returns:
        Dict with stats per content type and overall totals
    """
    results_by_type = {}
    total_processed = 0
    total_success = 0
    total_failed = 0

    # Process content types in explicit order
    processing_order = [
        ContentType.BLOCK,
        ContentType.STORE_AGENT,
        ContentType.DOCUMENTATION,
    ]

    for content_type in processing_order:
        handler = CONTENT_HANDLERS.get(content_type)
        if not handler:
            logger.warning(f"No handler registered for {content_type.value}")
            continue
        try:
            logger.info(f"Processing {content_type.value} content type...")

            # Get missing items from handler
            missing_items = await handler.get_missing_items(batch_size)

            if not missing_items:
                results_by_type[content_type.value] = {
                    "processed": 0,
                    "success": 0,
                    "failed": 0,
                    "message": "No missing embeddings",
                }
                continue

            # Process embeddings concurrently for better performance
            embedding_tasks = [
                ensure_content_embedding(
                    content_type=item.content_type,
                    content_id=item.content_id,
                    searchable_text=item.searchable_text,
                    metadata=item.metadata,
                    user_id=item.user_id,
                )
                for item in missing_items
            ]

            results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

            success = sum(1 for result in results if result is True)
            failed = len(results) - success

            # Aggregate unique errors to avoid Sentry spam
            if failed > 0:
                # Group errors by type and message
                error_summary: dict[str, int] = {}
                for result in results:
                    if isinstance(result, Exception):
                        error_key = f"{type(result).__name__}: {str(result)}"
                        error_summary[error_key] = error_summary.get(error_key, 0) + 1

                # Log aggregated error summary
                error_details = ", ".join(
                    f"{error} ({count}x)" for error, count in error_summary.items()
                )
                logger.error(
                    f"{content_type.value}: {failed}/{len(results)} embeddings failed. "
                    f"Errors: {error_details}"
                )

            results_by_type[content_type.value] = {
                "processed": len(missing_items),
                "success": success,
                "failed": failed,
                "message": f"Backfilled {success} embeddings, {failed} failed",
            }

            total_processed += len(missing_items)
            total_success += success
            total_failed += failed

            logger.info(
                f"{content_type.value}: processed {len(missing_items)}, "
                f"success {success}, failed {failed}"
            )

        except Exception as e:
            logger.error(f"Failed to process {content_type.value}: {e}")
            results_by_type[content_type.value] = {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "error": str(e),
            }

    return {
        "by_type": results_by_type,
        "totals": {
            "processed": total_processed,
            "success": total_success,
            "failed": total_failed,
            "message": f"Overall: {total_success} succeeded, {total_failed} failed",
        },
    }


async def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a search query.

    Same as generate_embedding but with clearer intent.
    Raises exceptions on failure - caller should handle.
    """
    return await generate_embedding(query)


def embedding_to_vector_string(embedding: list[float]) -> str:
    """Convert embedding list to PostgreSQL vector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def ensure_content_embedding(
    content_type: ContentType,
    content_id: str,
    searchable_text: str,
    metadata: dict | None = None,
    user_id: str | None = None,
    force: bool = False,
    tx: prisma.Prisma | None = None,
) -> bool:
    """
    Ensure an embedding exists for any content type.

    Generic function for creating embeddings for store agents, blocks, docs, etc.

    Args:
        content_type: ContentType enum value (STORE_AGENT, BLOCK, etc.)
        content_id: Unique identifier for the content
        searchable_text: Combined text for embedding generation
        metadata: Optional metadata to store with embedding
        force: Force regeneration even if embedding exists
        tx: Optional transaction client

    Returns:
        True if embedding exists/was created

    Raises exceptions on failure - caller should handle.
    """
    # Check if embedding already exists
    if not force:
        existing = await get_content_embedding(content_type, content_id, user_id)
        if existing and existing.get("embedding"):
            logger.debug(f"Embedding for {content_type}:{content_id} already exists")
            return True

    # Generate new embedding
    embedding = await generate_embedding(searchable_text)

    # Store the embedding
    return await store_content_embedding(
        content_type=content_type,
        content_id=content_id,
        embedding=embedding,
        searchable_text=searchable_text,
        metadata=metadata or {},
        user_id=user_id,
        tx=tx,
    )


async def cleanup_orphaned_embeddings() -> dict[str, Any]:
    """
    Clean up embeddings for content that no longer exists or is no longer valid.

    Compares current content with embeddings in database and removes orphaned records:
    - STORE_AGENT: Removes embeddings for rejected/deleted store listings
    - BLOCK: Removes embeddings for blocks no longer registered
    - DOCUMENTATION: Removes embeddings for deleted doc files

    Returns:
        Dict with cleanup statistics per content type
    """
    results_by_type = {}
    total_deleted = 0

    # Cleanup orphaned embeddings for all content types
    cleanup_types = [
        ContentType.STORE_AGENT,
        ContentType.BLOCK,
        ContentType.DOCUMENTATION,
    ]

    for content_type in cleanup_types:
        try:
            handler = CONTENT_HANDLERS.get(content_type)
            if not handler:
                logger.warning(f"No handler registered for {content_type}")
                results_by_type[content_type.value] = {
                    "deleted": 0,
                    "error": "No handler registered",
                }
                continue

            # Get all current content IDs from handler
            if content_type == ContentType.STORE_AGENT:
                # Get IDs of approved store listing versions from non-deleted listings
                valid_agents = await query_raw_with_schema(
                    """
                    SELECT slv.id
                    FROM {schema_prefix}"StoreListingVersion" slv
                    JOIN {schema_prefix}"StoreListing" sl ON slv."storeListingId" = sl.id
                    WHERE slv."submissionStatus" = 'APPROVED'
                      AND slv."isDeleted" = false
                      AND sl."isDeleted" = false
                    """,
                )
                current_ids = {row["id"] for row in valid_agents}
            elif content_type == ContentType.BLOCK:
                from backend.data.block import get_blocks

                current_ids = set(get_blocks().keys())
            elif content_type == ContentType.DOCUMENTATION:
                # Use DocumentationHandler to get section-based content IDs
                from backend.api.features.store.content_handlers import (
                    DocumentationHandler,
                )

                doc_handler = CONTENT_HANDLERS.get(ContentType.DOCUMENTATION)
                if isinstance(doc_handler, DocumentationHandler):
                    docs_root = doc_handler._get_docs_root()
                    if docs_root.exists():
                        current_ids = doc_handler._get_all_section_content_ids(
                            docs_root
                        )
                    else:
                        current_ids = set()
                else:
                    current_ids = set()
            else:
                # Skip unknown content types to avoid accidental deletion
                logger.warning(
                    f"Skipping cleanup for unknown content type: {content_type}"
                )
                results_by_type[content_type.value] = {
                    "deleted": 0,
                    "error": "Unknown content type - skipped for safety",
                }
                continue

            # Get all embedding IDs from database
            db_embeddings = await query_raw_with_schema(
                """
                SELECT "contentId"
                FROM {schema_prefix}"UnifiedContentEmbedding"
                WHERE "contentType" = $1::{schema_prefix}"ContentType"
                """,
                content_type,
            )

            db_ids = {row["contentId"] for row in db_embeddings}

            # Find orphaned embeddings (in DB but not in current content)
            orphaned_ids = db_ids - current_ids

            if not orphaned_ids:
                logger.info(f"{content_type.value}: No orphaned embeddings found")
                results_by_type[content_type.value] = {
                    "deleted": 0,
                    "message": "No orphaned embeddings",
                }
                continue

            # Delete orphaned embeddings in batch for better performance
            orphaned_list = list(orphaned_ids)
            try:
                await execute_raw_with_schema(
                    """
                    DELETE FROM {schema_prefix}"UnifiedContentEmbedding"
                    WHERE "contentType" = $1::{schema_prefix}"ContentType"
                      AND "contentId" = ANY($2::text[])
                    """,
                    content_type,
                    orphaned_list,
                )
                deleted = len(orphaned_list)
            except Exception as e:
                logger.error(f"Failed to batch delete orphaned embeddings: {e}")
                deleted = 0

            logger.info(
                f"{content_type.value}: Deleted {deleted}/{len(orphaned_ids)} orphaned embeddings"
            )
            results_by_type[content_type.value] = {
                "deleted": deleted,
                "orphaned": len(orphaned_ids),
                "message": f"Deleted {deleted} orphaned embeddings",
            }

            total_deleted += deleted

        except Exception as e:
            logger.error(f"Failed to cleanup {content_type.value}: {e}")
            results_by_type[content_type.value] = {
                "deleted": 0,
                "error": str(e),
            }

    return {
        "by_type": results_by_type,
        "totals": {
            "deleted": total_deleted,
            "message": f"Deleted {total_deleted} orphaned embeddings",
        },
    }


async def semantic_search(
    query: str,
    content_types: list[ContentType] | None = None,
    user_id: str | None = None,
    limit: int = 20,
    min_similarity: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Semantic search across content types using embeddings.

    Performs vector similarity search on UnifiedContentEmbedding table.
    Used directly for blocks/docs/library agents, or as the semantic component
    within hybrid_search for store agents.

    If embedding generation fails, falls back to lexical search on searchableText.

    Args:
        query: Search query string
        content_types: List of ContentType to search. Defaults to [BLOCK, STORE_AGENT, DOCUMENTATION]
        user_id: Optional user ID for searching private content (library agents)
        limit: Maximum number of results to return (default: 20)
        min_similarity: Minimum cosine similarity threshold (0-1, default: 0.5)

    Returns:
        List of search results with the following structure:
        [
            {
                "content_id": str,
                "content_type": str,  # "BLOCK", "STORE_AGENT", "DOCUMENTATION", or "LIBRARY_AGENT"
                "searchable_text": str,
                "metadata": dict,
                "similarity": float,  # Cosine similarity score (0-1)
            },
            ...
        ]

    Examples:
        # Search blocks only
        results = await semantic_search("calculate", content_types=[ContentType.BLOCK])

        # Search blocks and documentation
        results = await semantic_search(
            "how to use API",
            content_types=[ContentType.BLOCK, ContentType.DOCUMENTATION]
        )

        # Search all public content (default)
        results = await semantic_search("AI agent")

        # Search user's library agents
        results = await semantic_search(
            "my custom agent",
            content_types=[ContentType.LIBRARY_AGENT],
            user_id="user123"
        )
    """
    # Default to searching all public content types
    if content_types is None:
        content_types = [
            ContentType.BLOCK,
            ContentType.STORE_AGENT,
            ContentType.DOCUMENTATION,
        ]

    # Validate inputs
    if not content_types:
        return []  # Empty content_types would cause invalid SQL (IN ())

    query = query.strip()
    if not query:
        return []

    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    # Generate query embedding
    try:
        query_embedding = await embed_query(query)
        # Semantic search with embeddings
        embedding_str = embedding_to_vector_string(query_embedding)

        # Build params in order: limit, then user_id (if provided), then content types
        params: list[Any] = [limit]
        user_filter = ""
        if user_id is not None:
            user_filter = 'AND "userId" = ${}'.format(len(params) + 1)
            params.append(user_id)

        # Add content type parameters and build placeholders dynamically
        content_type_start_idx = len(params) + 1
        content_type_placeholders = ", ".join(
            "$" + str(content_type_start_idx + i) + '::{schema_prefix}"ContentType"'
            for i in range(len(content_types))
        )
        params.extend([ct.value for ct in content_types])

        # Build min_similarity param index before appending
        min_similarity_idx = len(params) + 1
        params.append(min_similarity)

        # Use unqualified ::vector and <=> operator - pgvector is in search_path on all environments
        sql = (
            """
            SELECT
                "contentId" as content_id,
                "contentType" as content_type,
                "searchableText" as searchable_text,
                metadata,
                1 - (embedding <=> '"""
            + embedding_str
            + """'::vector) as similarity
            FROM {schema_prefix}"UnifiedContentEmbedding"
            WHERE "contentType" IN ("""
            + content_type_placeholders
            + """)
            """
            + user_filter
            + """
            AND 1 - (embedding <=> '"""
            + embedding_str
            + """'::vector) >= $"""
            + str(min_similarity_idx)
            + """
            ORDER BY similarity DESC
            LIMIT $1
        """
        )

        results = await query_raw_with_schema(sql, *params)
        return [
            {
                "content_id": row["content_id"],
                "content_type": row["content_type"],
                "searchable_text": row["searchable_text"],
                "metadata": row["metadata"],
                "similarity": float(row["similarity"]),
            }
            for row in results
        ]
    except Exception as e:
        logger.warning(f"Semantic search failed, falling back to lexical search: {e}")

    # Fallback to lexical search if embeddings unavailable

    params_lexical: list[Any] = [limit]
    user_filter = ""
    if user_id is not None:
        user_filter = 'AND "userId" = ${}'.format(len(params_lexical) + 1)
        params_lexical.append(user_id)

    # Add content type parameters and build placeholders dynamically
    content_type_start_idx = len(params_lexical) + 1
    content_type_placeholders_lexical = ", ".join(
        "$" + str(content_type_start_idx + i) + '::{schema_prefix}"ContentType"'
        for i in range(len(content_types))
    )
    params_lexical.extend([ct.value for ct in content_types])

    # Build query param index before appending
    query_param_idx = len(params_lexical) + 1
    params_lexical.append(f"%{query}%")

    # Use regular string (not f-string) for template to preserve {schema_prefix} placeholders
    sql_lexical = (
        """
        SELECT
            "contentId" as content_id,
            "contentType" as content_type,
            "searchableText" as searchable_text,
            metadata,
            0.0 as similarity
        FROM {schema_prefix}"UnifiedContentEmbedding"
        WHERE "contentType" IN ("""
        + content_type_placeholders_lexical
        + """)
        """
        + user_filter
        + """
        AND "searchableText" ILIKE $"""
        + str(query_param_idx)
        + """
        ORDER BY "updatedAt" DESC
        LIMIT $1
    """
    )

    try:
        results = await query_raw_with_schema(sql_lexical, *params_lexical)
        return [
            {
                "content_id": row["content_id"],
                "content_type": row["content_type"],
                "searchable_text": row["searchable_text"],
                "metadata": row["metadata"],
                "similarity": 0.0,  # Lexical search doesn't provide similarity
            }
            for row in results
        ]
    except Exception as e:
        logger.error(f"Lexical search failed: {e}")
        return []
