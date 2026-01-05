"""
Hybrid Search for Store Agents

Combines semantic (embedding) search with lexical (tsvector) search
for improved relevance in marketplace agent discovery.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import prisma

from backend.api.features.store.embeddings import embed_query, embedding_to_vector_string

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchWeights:
    """Weights for combining search signals."""

    semantic: float = 0.35  # Embedding cosine similarity
    lexical: float = 0.35  # tsvector ts_rank_cd score
    category: float = 0.20  # Category match boost
    recency: float = 0.10  # Newer agents ranked higher


DEFAULT_WEIGHTS = HybridSearchWeights()

# Minimum relevance score threshold - agents below this are filtered out
# With weights (0.35 semantic + 0.35 lexical + 0.20 category + 0.10 recency):
# - 0.20 means at least ~50% semantic match OR strong lexical match required
# - Ensures only genuinely relevant results are returned
# - Recency alone (0.10 max) won't pass the threshold
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
    client = prisma.get_client()

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

    # Determine if we can use hybrid search (have query embedding)
    use_hybrid = query_embedding is not None

    if use_hybrid:
        # Add embedding parameter
        embedding_str = embedding_to_vector_string(query_embedding)
        params.append(embedding_str)
        embedding_param = f"${param_index}"
        param_index += 1

        # Build hybrid search query with weighted scoring
        # The semantic score is (1 - cosine_distance), normalized to [0,1]
        # The lexical score is ts_rank_cd, normalized by max value
        sql_query = f"""
            WITH search_scores AS (
                SELECT
                    sa.*,
                    -- Semantic score: cosine similarity (1 - distance)
                    COALESCE(1 - (sle.embedding <=> {embedding_param}::vector), 0) as semantic_score,
                    -- Lexical score: ts_rank_cd normalized
                    COALESCE(ts_rank_cd(sa.search, plainto_tsquery('english', {query_param})), 0) as lexical_raw,
                    -- Category match: 1 if query term appears in categories, else 0
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM unnest(sa.categories) cat
                            WHERE LOWER(cat) LIKE '%' || LOWER({query_param}) || '%'
                        ) THEN 1.0
                        ELSE 0.0
                    END as category_score,
                    -- Recency score: exponential decay over 90 days
                    EXP(-EXTRACT(EPOCH FROM (NOW() - sa.updated_at)) / (90 * 24 * 3600)) as recency_score
                FROM "StoreAgent" sa
                LEFT JOIN "StoreListing" sl ON sa.slug = sl.slug
                LEFT JOIN "StoreListingVersion" slv ON sl."activeVersionId" = slv.id
                LEFT JOIN "StoreListingEmbedding" sle ON slv.id = sle."storeListingVersionId"
                WHERE {where_clause}
                AND (
                    sa.search @@ plainto_tsquery('english', {query_param})
                    OR sle.embedding IS NOT NULL
                )
            ),
            normalized AS (
                SELECT
                    *,
                    -- Normalize lexical score by max in result set
                    CASE
                        WHEN MAX(lexical_raw) OVER () > 0
                        THEN lexical_raw / MAX(lexical_raw) OVER ()
                        ELSE 0
                    END as lexical_score
                FROM search_scores
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
                    (
                        {weights.semantic} * semantic_score +
                        {weights.lexical} * lexical_score +
                        {weights.category} * category_score +
                        {weights.recency} * recency_score
                    ) as combined_score
                FROM normalized
            )
            SELECT * FROM scored
            WHERE combined_score >= {min_score}
            ORDER BY combined_score DESC
            LIMIT ${param_index} OFFSET ${param_index + 1}
        """

        # Add pagination params
        params.extend([page_size, offset])

        # Count query - must also filter by min_score
        count_query = f"""
            WITH search_scores AS (
                SELECT
                    sa.slug,
                    COALESCE(1 - (sle.embedding <=> {embedding_param}::vector), 0) as semantic_score,
                    COALESCE(ts_rank_cd(sa.search, plainto_tsquery('english', {query_param})), 0) as lexical_raw,
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM unnest(sa.categories) cat
                            WHERE LOWER(cat) LIKE '%' || LOWER({query_param}) || '%'
                        ) THEN 1.0
                        ELSE 0.0
                    END as category_score,
                    EXP(-EXTRACT(EPOCH FROM (NOW() - sa.updated_at)) / (90 * 24 * 3600)) as recency_score
                FROM "StoreAgent" sa
                LEFT JOIN "StoreListing" sl ON sa.slug = sl.slug
                LEFT JOIN "StoreListingVersion" slv ON sl."activeVersionId" = slv.id
                LEFT JOIN "StoreListingEmbedding" sle ON slv.id = sle."storeListingVersionId"
                WHERE {where_clause}
                AND (
                    sa.search @@ plainto_tsquery('english', {query_param})
                    OR sle.embedding IS NOT NULL
                )
            ),
            normalized AS (
                SELECT
                    slug,
                    semantic_score,
                    category_score,
                    recency_score,
                    CASE
                        WHEN MAX(lexical_raw) OVER () > 0
                        THEN lexical_raw / MAX(lexical_raw) OVER ()
                        ELSE 0
                    END as lexical_score
                FROM search_scores
            ),
            scored AS (
                SELECT
                    slug,
                    (
                        {weights.semantic} * semantic_score +
                        {weights.lexical} * lexical_score +
                        {weights.category} * category_score +
                        {weights.recency} * recency_score
                    ) as combined_score
                FROM normalized
            )
            SELECT COUNT(*) as count FROM scored
            WHERE combined_score >= {min_score}
        """

    else:
        # Fallback to lexical-only search (existing behavior)
        # Note: For lexical-only, we still require tsvector match but don't
        # apply min_score since ts_rank_cd isn't normalized to [0,1]
        logger.warning("Falling back to lexical-only search (no query embedding)")

        sql_query = f"""
            WITH lexical_scores AS (
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
                    0.0 as semantic_score,
                    ts_rank_cd(search, plainto_tsquery('english', {query_param})) as lexical_raw,
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM unnest(categories) cat
                            WHERE LOWER(cat) LIKE '%' || LOWER({query_param}) || '%'
                        ) THEN 1.0
                        ELSE 0.0
                    END as category_score,
                    EXP(-EXTRACT(EPOCH FROM (NOW() - updated_at)) / (90 * 24 * 3600)) as recency_score
                FROM "StoreAgent" sa
                WHERE {where_clause}
                AND search @@ plainto_tsquery('english', {query_param})
            ),
            normalized AS (
                SELECT
                    *,
                    CASE
                        WHEN MAX(lexical_raw) OVER () > 0
                        THEN lexical_raw / MAX(lexical_raw) OVER ()
                        ELSE 0
                    END as lexical_score
                FROM lexical_scores
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
                    (
                        {weights.lexical} * lexical_score +
                        {weights.category} * category_score +
                        {weights.recency} * recency_score
                    ) as combined_score
                FROM normalized
            )
            SELECT * FROM scored
            WHERE combined_score >= {min_score}
            ORDER BY combined_score DESC
            LIMIT ${param_index} OFFSET ${param_index + 1}
        """

        params.extend([page_size, offset])

        count_query = f"""
            WITH lexical_scores AS (
                SELECT
                    slug,
                    ts_rank_cd(search, plainto_tsquery('english', {query_param})) as lexical_raw,
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM unnest(categories) cat
                            WHERE LOWER(cat) LIKE '%' || LOWER({query_param}) || '%'
                        ) THEN 1.0
                        ELSE 0.0
                    END as category_score,
                    EXP(-EXTRACT(EPOCH FROM (NOW() - updated_at)) / (90 * 24 * 3600)) as recency_score
                FROM "StoreAgent" sa
                WHERE {where_clause}
                AND search @@ plainto_tsquery('english', {query_param})
            ),
            normalized AS (
                SELECT
                    slug,
                    category_score,
                    recency_score,
                    CASE
                        WHEN MAX(lexical_raw) OVER () > 0
                        THEN lexical_raw / MAX(lexical_raw) OVER ()
                        ELSE 0
                    END as lexical_score
                FROM lexical_scores
            ),
            scored AS (
                SELECT
                    slug,
                    (
                        {weights.lexical} * lexical_score +
                        {weights.category} * category_score +
                        {weights.recency} * recency_score
                    ) as combined_score
                FROM normalized
            )
            SELECT COUNT(*) as count FROM scored
            WHERE combined_score >= {min_score}
        """

    try:
        # Execute search query
        # Dynamic SQL is safe here - all user inputs are parameterized ($1, $2, etc.)
        results = await client.query_raw(sql_query, *params)  # type: ignore[arg-type]

        # Execute count query (without pagination params)
        count_params = params[:-2]  # Remove LIMIT and OFFSET params
        count_result = await client.query_raw(count_query, *count_params)  # type: ignore[arg-type]
        total = count_result[0]["count"] if count_result else 0

        logger.info(
            f"Hybrid search for '{query}': {len(results)} results, {total} total "
            f"(hybrid={use_hybrid})"
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
