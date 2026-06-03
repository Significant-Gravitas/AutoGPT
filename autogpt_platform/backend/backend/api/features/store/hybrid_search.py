"""
Hybrid search for store agents.

Wraps the generic search engine in ``backend.api.features.search.hybrid_search``
with store-specific concerns: a ``StoreAgent``-joined SQL query that pulls
display metadata (runs, rating, featured, creator) and a weight schema
that adds a ``popularity`` signal.

For generic cross-content search use ``search.hybrid_search.unified_hybrid_search``.
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from backend.api.features.search.embeddings import (
    EMBEDDING_DIM,
    embed_query,
    embedding_to_vector_string,
)
from backend.api.features.search.hybrid_search import (
    DEFAULT_STORE_AGENT_MIN_SCORE,
    HybridSearchRow,
    _log_vector_error_diagnostics,
    bm25_rerank,
)
from backend.data.db import query_raw_with_schema

logger = logging.getLogger(__name__)


@dataclass
class StoreAgentSearchWeights:
    """Weights for store agent search including popularity."""

    semantic: float = 0.30
    lexical: float = 0.30
    category: float = 0.20
    recency: float = 0.10
    popularity: float = 0.10

    def __post_init__(self):
        total = (
            self.semantic
            + self.lexical
            + self.category
            + self.recency
            + self.popularity
        )
        if any(
            w < 0
            for w in [
                self.semantic,
                self.lexical,
                self.category,
                self.recency,
                self.popularity,
            ]
        ):
            raise ValueError("All weights must be non-negative")
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to ~1.0, got {total:.3f}")


DEFAULT_STORE_AGENT_WEIGHTS = StoreAgentSearchWeights()


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
    weights: StoreAgentSearchWeights | None = None,
    min_score: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Hybrid search for store agents with full metadata.

    Uses UnifiedContentEmbedding for search, joins to StoreAgent for metadata.
    """
    query = query.strip()
    if not query:
        return [], 0

    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 1
    if page_size > 100:
        page_size = 100

    if weights is None:
        weights = DEFAULT_STORE_AGENT_WEIGHTS
    if min_score is None:
        min_score = (
            DEFAULT_STORE_AGENT_MIN_SCORE  # Use original threshold for store agents
        )

    offset = (page - 1) * page_size

    # Generate query embedding with graceful degradation
    try:
        query_embedding = await embed_query(query)
    except Exception as e:
        logger.warning(
            f"Failed to generate query embedding - falling back to lexical-only search: {e}"
        )
        query_embedding = [0.0] * EMBEDDING_DIM
        total_non_semantic = (
            weights.lexical + weights.category + weights.recency + weights.popularity
        )
        if total_non_semantic > 0:
            factor = 1.0 / total_non_semantic
            weights = StoreAgentSearchWeights(
                semantic=0.0,
                lexical=weights.lexical * factor,
                category=weights.category * factor,
                recency=weights.recency * factor,
                popularity=weights.popularity * factor,
            )
        else:
            weights = StoreAgentSearchWeights(
                semantic=0.0, lexical=1.0, category=0.0, recency=0.0, popularity=0.0
            )

    # Build parameters
    params: list[Any] = []
    param_idx = 1

    params.append(query)
    query_param = f"${param_idx}"
    param_idx += 1

    params.append(query.lower())
    query_lower_param = f"${param_idx}"
    param_idx += 1

    embedding_str = embedding_to_vector_string(query_embedding)
    params.append(embedding_str)
    embedding_param = f"${param_idx}"
    param_idx += 1

    # Build WHERE clause for StoreAgent filters
    where_parts = ["sa.is_available = true"]

    if featured:
        where_parts.append("sa.featured = true")

    if creators:
        params.append(creators)
        where_parts.append(f"sa.creator_username = ANY(${param_idx})")
        param_idx += 1

    if category:
        params.append(category)
        where_parts.append(f"${param_idx} = ANY(sa.categories)")
        param_idx += 1

    where_clause = " AND ".join(where_parts)

    # Weights
    params.append(weights.semantic)
    w_semantic = f"${param_idx}"
    param_idx += 1

    params.append(weights.lexical)
    w_lexical = f"${param_idx}"
    param_idx += 1

    params.append(weights.category)
    w_category = f"${param_idx}"
    param_idx += 1

    params.append(weights.recency)
    w_recency = f"${param_idx}"
    param_idx += 1

    params.append(weights.popularity)
    w_popularity = f"${param_idx}"
    param_idx += 1

    params.append(min_score)
    min_score_param = f"${param_idx}"
    param_idx += 1

    params.append(page_size)
    limit_param = f"${param_idx}"
    param_idx += 1

    params.append(offset)
    offset_param = f"${param_idx}"
    param_idx += 1

    # Query using UnifiedContentEmbedding for search, StoreAgent for metadata
    sql_query = f"""
        WITH candidates AS (
            -- Lexical matches via UnifiedContentEmbedding.search
            SELECT uce."contentId" as "storeListingVersionId"
            FROM {{schema_prefix}}"UnifiedContentEmbedding" uce
            INNER JOIN {{schema_prefix}}"StoreAgent" sa
                ON uce."contentId" = sa.listing_version_id
            WHERE uce."contentType" = 'STORE_AGENT'::{{schema_prefix}}"ContentType"
            AND uce."userId" IS NULL
            AND uce.search @@ plainto_tsquery('english', {query_param})
            AND {where_clause}

            UNION

            -- Semantic matches via UnifiedContentEmbedding.embedding
            SELECT uce."contentId" as "storeListingVersionId"
            FROM (
                SELECT uce."contentId", uce.embedding
                FROM {{schema_prefix}}"UnifiedContentEmbedding" uce
                INNER JOIN {{schema_prefix}}"StoreAgent" sa
                    ON uce."contentId" = sa.listing_version_id
                WHERE uce."contentType" = 'STORE_AGENT'::{{schema_prefix}}"ContentType"
                AND uce."userId" IS NULL
                AND {where_clause}
                ORDER BY uce.embedding <=> {embedding_param}::vector
                LIMIT 200
            ) uce
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
                sa.graph_id,
                -- Searchable text for BM25 reranking
                COALESCE(sa.agent_name, '') || ' ' || COALESCE(sa.sub_heading, '') || ' ' || COALESCE(sa.description, '') as searchable_text,
                -- Semantic score
                COALESCE(1 - (uce.embedding <=> {embedding_param}::vector), 0) as semantic_score,
                -- Lexical score (raw, will normalize)
                COALESCE(ts_rank_cd(uce.search, plainto_tsquery('english', {query_param})), 0) as lexical_raw,
                -- Category match
                CASE
                    WHEN EXISTS (
                        SELECT 1 FROM unnest(sa.categories) cat
                        WHERE LOWER(cat) LIKE '%' || {query_lower_param} || '%'
                    )
                    THEN 1.0
                    ELSE 0.0
                END as category_score,
                -- Recency
                GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW() - sa.updated_at)) / (90 * 24 * 3600)) as recency_score,
                -- Popularity (raw)
                sa.runs as popularity_raw
            FROM candidates c
            INNER JOIN {{schema_prefix}}"StoreAgent" sa
                ON c."storeListingVersionId" = sa.listing_version_id
            INNER JOIN {{schema_prefix}}"UnifiedContentEmbedding" uce
                ON sa.listing_version_id = uce."contentId"
                AND uce."contentType" = 'STORE_AGENT'::{{schema_prefix}}"ContentType"
        ),
        max_vals AS (
            SELECT
                GREATEST(MAX(lexical_raw), 0.001) as max_lexical,
                GREATEST(MAX(popularity_raw), 1) as max_popularity
            FROM search_scores
        ),
        normalized AS (
            SELECT
                ss.*,
                ss.lexical_raw / mv.max_lexical as lexical_score,
                CASE
                    WHEN ss.popularity_raw > 0
                    THEN LN(1 + ss.popularity_raw) / LN(1 + mv.max_popularity)
                    ELSE 0
                END as popularity_score
            FROM search_scores ss
            CROSS JOIN max_vals mv
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
                graph_id,
                searchable_text,
                semantic_score,
                lexical_score,
                category_score,
                recency_score,
                popularity_score,
                (
                    {w_semantic} * semantic_score +
                    {w_lexical} * lexical_score +
                    {w_category} * category_score +
                    {w_recency} * recency_score +
                    {w_popularity} * popularity_score
                ) as combined_score
            FROM normalized
        ),
        filtered AS (
            SELECT *, COUNT(*) OVER () as total_count
            FROM scored
            WHERE combined_score >= {min_score_param}
        )
        SELECT * FROM filtered
        ORDER BY combined_score DESC
        LIMIT {limit_param} OFFSET {offset_param}
    """

    try:
        raw_results = await query_raw_with_schema(sql_query, *params)
    except Exception as e:
        await _log_vector_error_diagnostics(e)
        raise

    # Cast at the SQL boundary so the rest of the pipeline operates on
    # the typed row shape (matches search/hybrid_search.py's pattern).
    results: list[HybridSearchRow] = cast(list[HybridSearchRow], raw_results)
    total = results[0].get("total_count", 0) if results else 0

    # Apply BM25 reranking
    if results:
        results = bm25_rerank(
            query=query,
            results=results,
            text_field="searchable_text",
            bm25_weight=0.3,
            original_score_field="combined_score",
        )

    for result in results:
        result.pop("total_count", None)
        result.pop("searchable_text", None)

    logger.info(f"Hybrid search (store agents): {len(results)} results, {total} total")

    return cast(list[dict[str, Any]], results), total


async def hybrid_search_simple(
    query: str,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """Simplified hybrid search for store agents."""
    return await hybrid_search(query=query, page=page, page_size=page_size)


# Backward compatibility alias - HybridSearchWeights maps to StoreAgentSearchWeights
# for existing code that expects the popularity parameter
HybridSearchWeights = StoreAgentSearchWeights
