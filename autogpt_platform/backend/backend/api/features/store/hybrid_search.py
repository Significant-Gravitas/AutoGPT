"""
Hybrid Search for Store Agents

Combines semantic (embedding) search with lexical (tsvector) search
for improved relevance in marketplace agent discovery.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from backend.api.features.store.embeddings import (
    embed_query,
    embedding_to_vector_string,
)
from backend.data.db import query_raw_with_schema

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchWeights:
    """Weights for combining search signals."""

    semantic: float = 0.30  # Embedding cosine similarity
    lexical: float = 0.30  # tsvector ts_rank_cd score
    category: float = 0.20  # Category match boost
    recency: float = 0.10  # Newer agents ranked higher
    popularity: float = 0.10  # Agent usage/runs (PageRank-like)


DEFAULT_WEIGHTS = HybridSearchWeights()

# Minimum relevance score threshold - agents below this are filtered out
# With weights (0.30 semantic + 0.30 lexical + 0.20 category + 0.10 recency + 0.10 popularity):
# - 0.20 means at least ~60% semantic match OR strong lexical match required
# - Ensures only genuinely relevant results are returned
# - Recency/popularity alone (0.10 each) won't pass the threshold
DEFAULT_MIN_SCORE = 0.20


@dataclass
class HybridSearchResult:
    """A single search result with score breakdown."""

    slug: str
    agent_name: str
    agent_image: str
    creator_username: str
    creator_avatar: str
    sub_heading: str
    description: str
    runs: int
    rating: float
    categories: list[str]
    featured: bool
    is_available: bool
    updated_at: datetime

    # Score breakdown (for debugging/tuning)
    combined_score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    category_score: float = 0.0
    recency_score: float = 0.0
    popularity_score: float = 0.0


async def hybrid_search(
    query: str,
    featured: bool = False,
    creators: list[str] | None = None,
    category: str | None = None,
    sorted_by: (
        Literal["relevance", "rating", "runs", "name", "updated_at"] | None
    ) = None,
    page: int = 1,
    page_size: int = 20,
    weights: HybridSearchWeights | None = None,
    min_score: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Perform hybrid search combining semantic and lexical signals.

    Args:
        query: Search query string
        featured: Filter for featured agents only
        creators: Filter by creator usernames
        category: Filter by category
        sorted_by: Sort order (relevance uses hybrid scoring)
        page: Page number (1-indexed)
        page_size: Results per page
        weights: Custom weights for search signals
        min_score: Minimum relevance score threshold (0-1). Results below
                   this score are filtered out. Defaults to DEFAULT_MIN_SCORE.

    Returns:
        Tuple of (results list, total count). Returns empty list if no
        results meet the minimum relevance threshold.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if min_score is None:
        min_score = DEFAULT_MIN_SCORE

    offset = (page - 1) * page_size

    # Generate query embedding
    query_embedding = await embed_query(query)

    # Build WHERE clause conditions
    where_parts: list[str] = ["sa.is_available = true"]
    params: list[Any] = []
    param_index = 1

    # Add search query for lexical matching
    params.append(query)
    query_param = f"${param_index}"
    param_index += 1

    # Add lowercased query for category matching
    params.append(query.lower())
    query_lower_param = f"${param_index}"
    param_index += 1

    if featured:
        where_parts.append("sa.featured = true")

    if creators:
        where_parts.append(f"sa.creator_username = ANY(${param_index})")
        params.append(creators)
        param_index += 1

    if category:
        where_parts.append(f"${param_index} = ANY(sa.categories)")
        params.append(category)
        param_index += 1

    where_clause = " AND ".join(where_parts)

    # Embedding is required for hybrid search - fail fast if unavailable
    if query_embedding is None:
        raise ValueError(
            "Failed to generate query embedding. Hybrid search requires embeddings. "
            "Check that openai_internal_api_key is configured and OpenAI API is accessible."
        )

    # Add embedding parameter
    embedding_str = embedding_to_vector_string(query_embedding)
    params.append(embedding_str)
    embedding_param = f"${param_index}"
    param_index += 1

    # Optimized hybrid search query:
    # 1. Direct join to UnifiedContentEmbedding via contentId=storeListingVersionId (no redundant JOINs)
    # 2. UNION approach (deduplicates agents matching both branches)
    # 3. COUNT(*) OVER() to get total count in single query
    # 4. Optimized category matching with EXISTS + unnest
    # 5. Pre-calculated max values for lexical and popularity normalization
    # 6. Simplified recency calculation with linear decay
    # 7. Logarithmic popularity scaling to prevent viral agents from dominating
    sql_query = f"""
            WITH candidates AS (
                -- Lexical matches (uses GIN index on search column)
                SELECT sa."storeListingVersionId"
                FROM {{schema_prefix}}"StoreAgent" sa
                WHERE {where_clause}
                AND sa.search @@ plainto_tsquery('english', {query_param})

                UNION

                -- Semantic matches (uses HNSW index on embedding)
                SELECT sa."storeListingVersionId"
                FROM {{schema_prefix}}"StoreAgent" sa
                INNER JOIN {{schema_prefix}}"UnifiedContentEmbedding" uce
                    ON sa."storeListingVersionId" = uce."contentId" AND uce."contentType" = 'STORE_AGENT'
                WHERE {where_clause}
            ),
            search_scores AS (
                SELECT
                    sa.slug,
                    sa.agent_name,
                    sa.agent_image,
                    sa.creator_username,
                    sa.creator_avatar,
                    sa.sub_heading,
                    sa.description,
                    sa.runs,
                    sa.rating,
                    sa.categories,
                    sa.featured,
                    sa.is_available,
                    sa.updated_at,
                    -- Semantic score: cosine similarity (1 - distance)
                    COALESCE(1 - (uce.embedding <=> {embedding_param}::vector), 0) as semantic_score,
                    -- Lexical score: ts_rank_cd (will be normalized later)
                    COALESCE(ts_rank_cd(sa.search, plainto_tsquery('english', {query_param})), 0) as lexical_raw,
                    -- Category match: optimized with unnest for better performance
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM unnest(sa.categories) cat
                            WHERE LOWER(cat) LIKE '%' || {query_lower_param} || '%'
                        )
                        THEN 1.0
                        ELSE 0.0
                    END as category_score,
                    -- Recency score: linear decay over 90 days (simpler than exponential)
                    GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW() - sa.updated_at)) / (90 * 24 * 3600)) as recency_score,
                    -- Popularity raw: agent runs count (will be normalized with log scaling)
                    sa.runs as popularity_raw
                FROM candidates c
                INNER JOIN {{schema_prefix}}"StoreAgent" sa
                    ON c."storeListingVersionId" = sa."storeListingVersionId"
                LEFT JOIN {{schema_prefix}}"UnifiedContentEmbedding" uce
                    ON sa."storeListingVersionId" = uce."contentId" AND uce."contentType" = 'STORE_AGENT'
            ),
            max_lexical AS (
                SELECT MAX(lexical_raw) as max_val FROM search_scores
            ),
            max_popularity AS (
                SELECT MAX(popularity_raw) as max_val FROM search_scores
            ),
            normalized AS (
                SELECT
                    ss.*,
                    -- Normalize lexical score by pre-calculated max
                    CASE
                        WHEN ml.max_val > 0
                        THEN ss.lexical_raw / ml.max_val
                        ELSE 0
                    END as lexical_score,
                    -- Normalize popularity with logarithmic scaling to prevent viral agents from dominating
                    -- LOG(1 + runs) / LOG(1 + max_runs) ensures score is 0-1 range
                    CASE
                        WHEN mp.max_val > 0 AND ss.popularity_raw > 0
                        THEN LN(1 + ss.popularity_raw) / LN(1 + mp.max_val)
                        ELSE 0
                    END as popularity_score
                FROM search_scores ss
                CROSS JOIN max_lexical ml
                CROSS JOIN max_popularity mp
            ),
            scored AS (
                SELECT
                    slug,
                    agent_name,
                    agent_image,
                    creator_username,
                    creator_avatar,
                    sub_heading,
                    description,
                    runs,
                    rating,
                    categories,
                    featured,
                    is_available,
                    updated_at,
                    semantic_score,
                    lexical_score,
                    category_score,
                    recency_score,
                    popularity_score,
                    (
                        {weights.semantic} * semantic_score +
                        {weights.lexical} * lexical_score +
                        {weights.category} * category_score +
                        {weights.recency} * recency_score +
                        {weights.popularity} * popularity_score
                    ) as combined_score
                FROM normalized
            ),
            filtered AS (
                SELECT
                    *,
                    COUNT(*) OVER () as total_count
                FROM scored
                WHERE combined_score >= {min_score}
            )
            SELECT * FROM filtered
            ORDER BY combined_score DESC
            LIMIT ${param_index} OFFSET ${param_index + 1}
    """

    # Add pagination params
    params.extend([page_size, offset])

    try:
        # Execute search query - includes total_count via window function
        results = await query_raw_with_schema(sql_query, *params)

        # Extract total count from first result (all rows have same count)
        total = results[0]["total_count"] if results else 0

        # Remove total_count from results before returning
        for result in results:
            result.pop("total_count", None)

        logger.info(
            f"Hybrid search for '{query}': {len(results)} results, {total} total"
        )

        return results, total

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise


async def hybrid_search_simple(
    query: str,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """
    Simplified hybrid search for common use cases.

    Uses default weights and no filters.
    """
    return await hybrid_search(
        query=query,
        page=page,
        page_size=page_size,
    )
