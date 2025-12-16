"""
Store Listing Embeddings Service

Handles generation and storage of OpenAI embeddings for store listings
to enable semantic/hybrid search.
"""

import hashlib
import logging
import os
from typing import Any

import prisma

logger = logging.getLogger(__name__)

# OpenAI embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


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


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of text for change detection."""
    return hashlib.md5(text.encode()).hexdigest()


async def generate_embedding(text: str) -> list[float] | None:
    """
    Generate embedding for text using OpenAI API.

    Returns None if embedding generation fails.
    """
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, cannot generate embedding")
            return None

        client = OpenAI(api_key=api_key)

        # Truncate text to avoid token limits (~32k chars for safety)
        truncated_text = text[:32000]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=truncated_text,
        )

        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding with {len(embedding)} dimensions")
        return embedding

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None


async def store_embedding(
    version_id: str,
    embedding: list[float],
    searchable_text: str,
    content_hash: str,
    tx: prisma.Prisma | None = None,
) -> bool:
    """
    Store embedding in the database.

    Uses raw SQL since Prisma doesn't natively support pgvector.
    """
    try:
        client = tx if tx else prisma.get_client()

        # Convert embedding to PostgreSQL vector format
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # Upsert the embedding
        await client.execute_raw(
            """
            INSERT INTO "StoreListingEmbedding" (
                "id", "storeListingVersionId", "embedding",
                "searchableText", "contentHash", "createdAt", "updatedAt"
            )
            VALUES (
                gen_random_uuid(), $1, $2::vector,
                $3, $4, NOW(), NOW()
            )
            ON CONFLICT ("storeListingVersionId")
            DO UPDATE SET
                "embedding" = $2::vector,
                "searchableText" = $3,
                "contentHash" = $4,
                "updatedAt" = NOW()
            """,
            version_id,
            embedding_str,
            searchable_text,
            content_hash,
        )

        logger.info(f"Stored embedding for version {version_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to store embedding for version {version_id}: {e}")
        return False


async def get_embedding(version_id: str) -> dict[str, Any] | None:
    """
    Retrieve embedding record for a listing version.

    Returns dict with embedding, searchableText, contentHash or None if not found.
    """
    try:
        client = prisma.get_client()

        result = await client.query_raw(
            """
            SELECT
                "id",
                "storeListingVersionId",
                "embedding"::text as "embedding",
                "searchableText",
                "contentHash",
                "createdAt",
                "updatedAt"
            FROM "StoreListingEmbedding"
            WHERE "storeListingVersionId" = $1
            """,
            version_id,
        )

        if result and len(result) > 0:
            return result[0]
        return None

    except Exception as e:
        logger.error(f"Failed to get embedding for version {version_id}: {e}")
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

    Creates embedding if missing or if content has changed.
    Skips if content hash matches existing embedding.

    Args:
        version_id: The StoreListingVersion ID
        name: Agent name
        description: Agent description
        sub_heading: Agent sub-heading
        categories: Agent categories
        force: Force regeneration even if hash matches
        tx: Optional transaction client

    Returns:
        True if embedding exists/was created, False on failure
    """
    try:
        # Build searchable text and compute hash
        searchable_text = build_searchable_text(
            name, description, sub_heading, categories
        )
        content_hash = compute_content_hash(searchable_text)

        # Check if embedding already exists with same hash
        if not force:
            existing = await get_embedding(version_id)
            if existing and existing.get("contentHash") == content_hash:
                logger.debug(
                    f"Embedding for version {version_id} is up to date (hash match)"
                )
                return True

        # Generate new embedding
        embedding = await generate_embedding(searchable_text)
        if embedding is None:
            logger.warning(f"Could not generate embedding for version {version_id}")
            return False

        # Store the embedding
        return await store_embedding(
            version_id=version_id,
            embedding=embedding,
            searchable_text=searchable_text,
            content_hash=content_hash,
            tx=tx,
        )

    except Exception as e:
        logger.error(f"Failed to ensure embedding for version {version_id}: {e}")
        return False


async def delete_embedding(version_id: str) -> bool:
    """
    Delete embedding for a listing version.

    Note: This is usually handled automatically by CASCADE delete,
    but provided for manual cleanup if needed.
    """
    try:
        client = prisma.get_client()

        await client.execute_raw(
            """
            DELETE FROM "StoreListingEmbedding"
            WHERE "storeListingVersionId" = $1
            """,
            version_id,
        )

        logger.info(f"Deleted embedding for version {version_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete embedding for version {version_id}: {e}")
        return False


async def get_embedding_stats() -> dict[str, Any]:
    """
    Get statistics about embedding coverage.

    Returns counts of:
    - Total approved listing versions
    - Versions with embeddings
    - Versions without embeddings
    """
    try:
        client = prisma.get_client()

        # Count approved versions
        approved_result = await client.query_raw(
            """
            SELECT COUNT(*) as count
            FROM "StoreListingVersion"
            WHERE "submissionStatus" = 'APPROVED'
            AND "isDeleted" = false
            """
        )
        total_approved = approved_result[0]["count"] if approved_result else 0

        # Count versions with embeddings
        embedded_result = await client.query_raw(
            """
            SELECT COUNT(*) as count
            FROM "StoreListingVersion" slv
            JOIN "StoreListingEmbedding" sle ON slv.id = sle."storeListingVersionId"
            WHERE slv."submissionStatus" = 'APPROVED'
            AND slv."isDeleted" = false
            """
        )
        with_embeddings = embedded_result[0]["count"] if embedded_result else 0

        return {
            "total_approved": total_approved,
            "with_embeddings": with_embeddings,
            "without_embeddings": total_approved - with_embeddings,
            "coverage_percent": (
                round(with_embeddings / total_approved * 100, 1)
                if total_approved > 0
                else 0
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        return {
            "total_approved": 0,
            "with_embeddings": 0,
            "without_embeddings": 0,
            "coverage_percent": 0,
            "error": str(e),
        }


async def backfill_missing_embeddings(batch_size: int = 10) -> dict[str, Any]:
    """
    Generate embeddings for approved listings that don't have them.

    Args:
        batch_size: Number of embeddings to generate in one call

    Returns:
        Dict with success/failure counts
    """
    try:
        client = prisma.get_client()

        # Find approved versions without embeddings
        missing = await client.query_raw(
            """
            SELECT
                slv.id,
                slv.name,
                slv.description,
                slv."subHeading",
                slv.categories
            FROM "StoreListingVersion" slv
            LEFT JOIN "StoreListingEmbedding" sle ON slv.id = sle."storeListingVersionId"
            WHERE slv."submissionStatus" = 'APPROVED'
            AND slv."isDeleted" = false
            AND sle.id IS NULL
            LIMIT $1
            """,
            batch_size,
        )

        if not missing:
            return {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "message": "No missing embeddings",
            }

        success = 0
        failed = 0

        for row in missing:
            result = await ensure_embedding(
                version_id=row["id"],
                name=row["name"],
                description=row["description"],
                sub_heading=row["subHeading"],
                categories=row["categories"] or [],
            )
            if result:
                success += 1
            else:
                failed += 1

        return {
            "processed": len(missing),
            "success": success,
            "failed": failed,
            "message": f"Backfilled {success} embeddings, {failed} failed",
        }

    except Exception as e:
        logger.error(f"Failed to backfill embeddings: {e}")
        return {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "error": str(e),
        }


async def embed_query(query: str) -> list[float] | None:
    """
    Generate embedding for a search query.

    Same as generate_embedding but with clearer intent.
    """
    return await generate_embedding(query)


def embedding_to_vector_string(embedding: list[float]) -> str:
    """Convert embedding list to PostgreSQL vector string format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"
