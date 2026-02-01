"""
Unified Hybrid Search

Combines semantic (embedding) search with lexical (tsvector) search
for improved relevance across all content types (agents, blocks, docs).
Includes BM25 reranking for improved lexical relevance.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from prisma.enums import ContentType
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from backend.api.features.store.embeddings import (
    EMBEDDING_DIM,
    embed_query,
    embedding_to_vector_string,
)
from backend.data.db import query_raw_with_schema

logger = logging.getLogger(__name__)


# ============================================================================
# BM25 Reranking
# ============================================================================


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 - lowercase and split on non-alphanumeric."""
    if not text:
        return []
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def bm25_rerank(
    query: str,
    results: list[dict[str, Any]],
    text_field: str = "searchable_text",
    bm25_weight: float = 0.3,
    original_score_field: str = "combined_score",
) -> list[dict[str, Any]]:
    """
    Rerank search results using BM25.

    Combines the original combined_score with BM25 score for improved
    lexical relevance, especially for exact term matches.

    Args:
        query: The search query
        results: List of result dicts with text_field and original_score_field
        text_field: Field name containing the text to score
        bm25_weight: Weight for BM25 score (0-1). Original score gets (1 - bm25_weight)
        original_score_field: Field name containing the original score

    Returns:
        Results list sorted by combined score (BM25 + original)
    """
    if not results or not query:
        return results

    # Extract texts and tokenize
    corpus = [tokenize(r.get(text_field, "") or "") for r in results]

    # Handle edge case where all documents are empty
    if all(len(doc) == 0 for doc in corpus):
        return results

    # Build BM25 index
    bm25 = BM25Okapi(corpus)

    # Score query against corpus
    query_tokens = tokenize(query)
    if not query_tokens:
        return results

    bm25_scores = bm25.get_scores(query_tokens)

    # Normalize BM25 scores to 0-1 range
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    normalized_bm25 = [s / max_bm25 for s in bm25_scores]

    # Combine scores
    original_weight = 1.0 - bm25_weight
    for i, result in enumerate(results):
        original_score = result.get(original_score_field, 0) or 0
        result["bm25_score"] = normalized_bm25[i]
        final_score = (
            original_weight * original_score + bm25_weight * normalized_bm25[i]
        )
        result["final_score"] = final_score
        result["relevance"] = final_score

    # Sort by relevance descending
    results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

    return results


@dataclass
class UnifiedSearchWeights:
    """Weights for unified search (no popularity signal)."""

    semantic: float = 0.40  # Embedding cosine similarity
    lexical: float = 0.40  # tsvector ts_rank_cd score
    category: float = 0.10  # Category match boost (for types that have categories)
    recency: float = 0.10  # Newer content ranked higher

    def __post_init__(self):
        """Validate weights are non-negative and sum to approximately 1.0."""
        total = self.semantic + self.lexical + self.category + self.recency

        if any(
            w < 0 for w in [self.semantic, self.lexical, self.category, self.recency]
        ):
            raise ValueError("All weights must be non-negative")

        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to ~1.0, got {total:.3f}")


# Default weights for unified search
DEFAULT_UNIFIED_WEIGHTS = UnifiedSearchWeights()

# Minimum relevance score thresholds
DEFAULT_MIN_SCORE = 0.15  # For unified search (more permissive)
DEFAULT_STORE_AGENT_MIN_SCORE = 0.20  # For store agent search (original threshold)


async def unified_hybrid_search(
    query: str,
    content_types: list[ContentType] | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
    weights: UnifiedSearchWeights | None = None,
    min_score: float | None = None,
    user_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Unified hybrid search across all content types.

    Searches UnifiedContentEmbedding using both semantic (vector) and lexical (tsvector) signals.

    Args:
        query: Search query string
        content_types: List of content types to search. Defaults to all public types.
        category: Filter by category (for content types that support it)
        page: Page number (1-indexed)
        page_size: Results per page
        weights: Custom weights for search signals
        min_score: Minimum relevance score threshold (0-1)
        user_id: User ID for searching private content (library agents)

    Returns:
        Tuple of (results list, total count)
    """
    # Validate inputs
    query = query.strip()
    if not query:
        return [], 0

    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 1
    if page_size > 100:
        page_size = 100

    if content_types is None:
        content_types = [
            ContentType.STORE_AGENT,
            ContentType.BLOCK,
            ContentType.DOCUMENTATION,
        ]

    if weights is None:
        weights = DEFAULT_UNIFIED_WEIGHTS
    if min_score is None:
        min_score = DEFAULT_MIN_SCORE

    offset = (page - 1) * page_size

    # Generate query embedding with graceful degradation
    try:
        query_embedding = await embed_query(query)
    except Exception as e:
        logger.warning(
            f"Failed to generate query embedding - falling back to lexical-only search: {e}. "
            "Check that openai_internal_api_key is configured and OpenAI API is accessible."
        )
        query_embedding = [0.0] * EMBEDDING_DIM
        # Redistribute semantic weight to lexical
        total_non_semantic = weights.lexical + weights.category + weights.recency
        if total_non_semantic > 0:
            factor = 1.0 / total_non_semantic
            weights = UnifiedSearchWeights(
                semantic=0.0,
                lexical=weights.lexical * factor,
                category=weights.category * factor,
                recency=weights.recency * factor,
            )
        else:
            weights = UnifiedSearchWeights(
                semantic=0.0, lexical=1.0, category=0.0, recency=0.0
            )

    # Build parameters
    params: list[Any] = []
    param_idx = 1

    # Query for lexical search
    params.append(query)
    query_param = f"${param_idx}"
    param_idx += 1

    # Query lowercase for category matching
    params.append(query.lower())
    query_lower_param = f"${param_idx}"
    param_idx += 1

    # Embedding
    embedding_str = embedding_to_vector_string(query_embedding)
    params.append(embedding_str)
    embedding_param = f"${param_idx}"
    param_idx += 1

    # Content types
    content_type_values = [ct.value for ct in content_types]
    params.append(content_type_values)
    content_types_param = f"${param_idx}"
    param_idx += 1

    # User ID filter (for private content)
    user_filter = ""
    if user_id is not None:
        params.append(user_id)
        user_filter = f'AND (uce."userId" = ${param_idx} OR uce."userId" IS NULL)'
        param_idx += 1
    else:
        user_filter = 'AND uce."userId" IS NULL'

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

    # Min score
    params.append(min_score)
    min_score_param = f"${param_idx}"
    param_idx += 1

    # Pagination
    params.append(page_size)
    limit_param = f"${param_idx}"
    param_idx += 1

    params.append(offset)
    offset_param = f"${param_idx}"
    param_idx += 1

    # Unified search query on UnifiedContentEmbedding
    sql_query = f"""
        WITH candidates AS (
            -- Lexical matches (uses GIN index on search column)
            SELECT uce.id, uce."contentType", uce."contentId"
            FROM {{schema_prefix}}"UnifiedContentEmbedding" uce
            WHERE uce."contentType" = ANY({content_types_param}::{{schema_prefix}}"ContentType"[])
            {user_filter}
            AND uce.search @@ plainto_tsquery('english', {query_param})

            UNION

            -- Semantic matches (uses HNSW index on embedding)
            (
                SELECT uce.id, uce."contentType", uce."contentId"
                FROM {{schema_prefix}}"UnifiedContentEmbedding" uce
                WHERE uce."contentType" = ANY({content_types_param}::{{schema_prefix}}"ContentType"[])
                {user_filter}
                ORDER BY uce.embedding <=> {embedding_param}::vector
                LIMIT 200
            )
        ),
        search_scores AS (
            SELECT
                uce."contentType" as content_type,
                uce."contentId" as content_id,
                uce."searchableText" as searchable_text,
                uce.metadata,
                uce."updatedAt" as updated_at,
                -- Semantic score: cosine similarity (1 - distance)
                COALESCE(1 - (uce.embedding <=> {embedding_param}::vector), 0) as semantic_score,
                -- Lexical score: ts_rank_cd
                COALESCE(ts_rank_cd(uce.search, plainto_tsquery('english', {query_param})), 0) as lexical_raw,
                -- Category match from metadata
                CASE
                    WHEN uce.metadata ? 'categories' AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(uce.metadata->'categories') cat
                        WHERE LOWER(cat) LIKE '%' || {query_lower_param} || '%'
                    )
                    THEN 1.0
                    ELSE 0.0
                END as category_score,
                -- Recency score: linear decay over 90 days
                GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW() - uce."updatedAt")) / (90 * 24 * 3600)) as recency_score
            FROM candidates c
            INNER JOIN {{schema_prefix}}"UnifiedContentEmbedding" uce ON c.id = uce.id
        ),
        max_lexical AS (
            SELECT GREATEST(MAX(lexical_raw), 0.001) as max_val FROM search_scores
        ),
        normalized AS (
            SELECT
                ss.*,
                ss.lexical_raw / ml.max_val as lexical_score
            FROM search_scores ss
            CROSS JOIN max_lexical ml
        ),
        scored AS (
            SELECT
                content_type,
                content_id,
                searchable_text,
                metadata,
                updated_at,
                semantic_score,
                lexical_score,
                category_score,
                recency_score,
                (
                    {w_semantic} * semantic_score +
                    {w_lexical} * lexical_score +
                    {w_category} * category_score +
                    {w_recency} * recency_score
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

    results = await query_raw_with_schema(sql_query, *params)

    total = results[0]["total_count"] if results else 0
    # Apply BM25 reranking
    if results:
        results = bm25_rerank(
            query=query,
            results=results,
            text_field="searchable_text",
            bm25_weight=0.3,
            original_score_field="combined_score",
        )

    # Clean up results
    for result in results:
        result.pop("total_count", None)

    logger.info(f"Unified hybrid search: {len(results)} results, {total} total")

    return results, total


# ============================================================================
# Store Agent specific search (with full metadata)
# ============================================================================


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
                ON uce."contentId" = sa."storeListingVersionId"
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
                    ON uce."contentId" = sa."storeListingVersionId"
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
                sa."agentGraphId",
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
                ON c."storeListingVersionId" = sa."storeListingVersionId"
            INNER JOIN {{schema_prefix}}"UnifiedContentEmbedding" uce
                ON sa."storeListingVersionId" = uce."contentId"
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
                "agentGraphId",
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

    results = await query_raw_with_schema(sql_query, *params)

    total = results[0]["total_count"] if results else 0

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

    return results, total


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
