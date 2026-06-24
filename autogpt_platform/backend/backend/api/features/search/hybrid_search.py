"""
Unified Hybrid Search — shared engine across features.

Combines semantic (embedding) search with lexical (tsvector) search plus
BM25 reranking. Searches the ``UnifiedContentEmbedding`` table across all
content types (store agents, library agents, workspace files, chat
sessions, blocks, docs).

Feature-specific search functions (e.g. ``store.hybrid_search`` which
joins to ``StoreAgent`` for popularity/featured metadata) live next to
the feature they belong to and call into the helpers here.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from prisma.enums import ContentType
from rank_bm25 import BM25Okapi

# Source TypedDict/NotRequired from typing_extensions: pydantic's TypeAdapter
# cannot build a core schema for a typing.TypedDict on Python < 3.12 (the
# __orig_bases__ attribute it relies on was only added to typing.TypedDict in
# 3.12). The search RPC contract test validates HybridSearchRow via TypeAdapter.
from typing_extensions import NotRequired, TypedDict

from backend.api.features.search.embeddings import (
    EMBEDDING_DIM,
    embed_query,
    embedding_to_vector_string,
)
from backend.data.db import query_raw_with_schema


class HybridSearchRow(TypedDict):
    """One row from :func:`unified_hybrid_search`.

    The SQL CTE always populates the core columns
    (``content_id``/``content_type``/``searchable_text``/``metadata``/
    ``updated_at`` and the score breakdown). Optional fields are either
    attached by post-processing (``bm25_rerank``) or stripped before
    the row leaves this module (``total_count``).
    """

    content_type: ContentType | str
    content_id: str
    searchable_text: str
    metadata: dict[str, Any]
    updated_at: datetime | None
    semantic_score: float
    lexical_score: float
    category_score: float
    recency_score: float
    combined_score: float
    # Window-function side car, identical across all rows in one
    # response. Popped before the row leaves ``unified_hybrid_search``.
    total_count: NotRequired[int]
    # Added by ``bm25_rerank`` after the SQL roundtrip — absent when
    # rerank short-circuits (empty corpus / empty query tokens).
    bm25_score: NotRequired[float]
    final_score: NotRequired[float]
    relevance: NotRequired[float]


logger = logging.getLogger(__name__)


# ============================================================================
# BM25 Reranking
# ============================================================================


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def bm25_rerank(
    query: str,
    results: list[HybridSearchRow],
    text_field: str = "searchable_text",
    bm25_weight: float = 0.3,
    original_score_field: str = "combined_score",
) -> list[HybridSearchRow]:
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
    lexical_query: str | None = None,
    prefix_match: bool = False,
) -> tuple[list[HybridSearchRow], int]:
    """
    Unified hybrid search across all content types.

    Searches UnifiedContentEmbedding using both semantic (vector) and lexical (tsvector) signals.

    Args:
        query: Search query string (used for embedding + lexical unless
            ``lexical_query`` is provided).
        content_types: List of content types to search. Defaults to all public types.
        category: Filter by category (for content types that support it)
        page: Page number (1-indexed)
        page_size: Results per page
        weights: Custom weights for search signals
        min_score: Minimum relevance score threshold (0-1)
        user_id: User ID for searching private content (library agents)
        lexical_query: Optional override for the tsvector ``@@`` candidate
            selection and ``ts_rank_cd`` scoring. Use when the natural-
            language query is too long for ``plainto_tsquery``'s
            AND-of-terms semantics — typically a keyword-extracted form
            of ``query`` (e.g. ``"youtube summarize video"`` for a goal
            like ``"summarize a YouTube video..."``). Defaults to
            ``query`` when None.

    Returns:
        Tuple of (results list, total count)
    """
    # Validate inputs
    query = query.strip()
    if not query:
        return [], 0
    # Default the lexical query to the semantic query for backwards-
    # compatibility; callers (e.g. library similarity search) override
    # when ``plainto_tsquery``'s AND semantics over a long natural goal
    # would zero out every match.
    lexical_query = (lexical_query or query).strip() or query

    # Lexical matching mode. ``plainto_tsquery`` matches whole words only,
    # which is wrong for search-as-you-type callers (e.g. /search/global):
    # the last token is usually a partial word ("se", "fir"). With
    # ``prefix_match`` we build a prefix ``to_tsquery`` (``term:*``) from the
    # query's word tokens so partial words still match. ``to_tsquery`` drops
    # stopwords gracefully, so an all-stopword fragment simply yields no
    # lexical hits and falls back to the semantic signal.
    lexical_fn = "plainto_tsquery"
    lexical_tsquery = lexical_query
    if prefix_match:
        tokens = tokenize(lexical_query)
        if tokens:
            lexical_fn = "to_tsquery"
            lexical_tsquery = " & ".join(f"{t}:*" for t in tokens)

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

    # Query for lexical search (may differ from semantic ``query`` —
    # see the ``lexical_query`` parameter doc). Carries the prefix
    # ``to_tsquery`` form when ``prefix_match`` is set.
    params.append(lexical_tsquery)
    query_param = f"${param_idx}"
    param_idx += 1

    # Query lowercase for category matching — keep tied to the
    # natural-language ``query`` so categories still match the user's
    # intent words, not the stripped lexical form.
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

    # User ID filter (for private content).
    # Defense-in-depth: a per-user row with NULL userId would be matched by
    # the public ``OR userId IS NULL`` branch and leak across users. All of
    # LIBRARY_AGENT, WORKSPACE_FILE, and CHAT_SESSION are always per-user;
    # explicitly exclude any of those combined with a NULL userId. The DB
    # CHECK constraint in migration ``20260526120000`` makes that
    # impossible at write time, but this query-side guard keeps the
    # invariant local to the search code too.
    _USER_SCOPED_TYPES_SQL = (
        "('LIBRARY_AGENT'::{schema_prefix}\"ContentType\", "
        "'WORKSPACE_FILE'::{schema_prefix}\"ContentType\", "
        "'CHAT_SESSION'::{schema_prefix}\"ContentType\")"
    )
    user_filter = ""
    if user_id is not None:
        params.append(user_id)
        user_filter = (
            f'AND (uce."userId" = ${param_idx} OR uce."userId" IS NULL) '
            f'AND NOT (uce."contentType" IN {_USER_SCOPED_TYPES_SQL} '
            f'AND uce."userId" IS NULL)'
        )
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
            -- Bounded to 500 rows so a broad prefix like ``a:*`` can't
            -- drag the whole table into ts_rank_cd scoring. Matches the
            -- semantic branch's bounded design.
            (
                SELECT uce.id, uce."contentType", uce."contentId"
                FROM {{schema_prefix}}"UnifiedContentEmbedding" uce
                WHERE uce."contentType" = ANY({content_types_param}::{{schema_prefix}}"ContentType"[])
                {user_filter}
                AND uce.search @@ {lexical_fn}('english', {query_param})
                LIMIT 500
            )

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
                COALESCE(ts_rank_cd(uce.search, {lexical_fn}('english', {query_param})), 0) as lexical_raw,
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
        -- ``content_id`` is the deterministic tiebreaker: when several
        -- rows tie on ``combined_score`` (common with identical
        -- embeddings or repeated text), Postgres is free to return any
        -- order absent a secondary key, which lets the same row land on
        -- multiple pages under ``LIMIT/OFFSET`` paging.
        ORDER BY combined_score DESC, content_id ASC
        LIMIT {limit_param} OFFSET {offset_param}
    """

    try:
        raw_results = await query_raw_with_schema(sql_query, *params)
    except Exception as e:
        await _log_vector_error_diagnostics(e)
        raise

    # ``query_raw_with_schema`` returns ``list[dict[str, Any]]`` — the
    # SQL columns align with the ``HybridSearchRow`` shape, so a single
    # cast at the boundary lets the rest of the pipeline operate on a
    # typed row without per-call ``# type: ignore``s.
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

    # Clean up results
    for result in results:
        result.pop("total_count", None)

    logger.info(f"Unified hybrid search: {len(results)} results, {total} total")

    return results, total


# ============================================================================
# Diagnostics
# ============================================================================

# Rate limit: only log vector error diagnostics once per this interval
_VECTOR_DIAG_INTERVAL_SECONDS = 60
_last_vector_diag_time: float = 0


async def _log_vector_error_diagnostics(error: Exception) -> None:
    """Log diagnostic info when 'type vector does not exist' error occurs.

    Note: Diagnostic queries use query_raw_with_schema which may run on a different
    pooled connection than the one that failed. Session-level search_path can differ,
    so these diagnostics show cluster-wide state, not necessarily the failed session.

    Includes rate limiting to avoid log spam - only logs once per minute.
    Caller should re-raise the error after calling this function.
    """
    global _last_vector_diag_time

    # Check if this is the vector type error
    error_str = str(error).lower()
    if not (
        "type" in error_str and "vector" in error_str and "does not exist" in error_str
    ):
        return

    # Rate limit: only log once per interval
    now = time.time()
    if now - _last_vector_diag_time < _VECTOR_DIAG_INTERVAL_SECONDS:
        return
    _last_vector_diag_time = now

    try:
        diagnostics: dict[str, object] = {}

        try:
            search_path_result = await query_raw_with_schema("SHOW search_path")
            diagnostics["search_path"] = search_path_result
        except Exception as e:
            diagnostics["search_path"] = f"Error: {e}"

        try:
            schema_result = await query_raw_with_schema("SELECT current_schema()")
            diagnostics["current_schema"] = schema_result
        except Exception as e:
            diagnostics["current_schema"] = f"Error: {e}"

        try:
            user_result = await query_raw_with_schema(
                "SELECT current_user, session_user, current_database()"
            )
            diagnostics["user_info"] = user_result
        except Exception as e:
            diagnostics["user_info"] = f"Error: {e}"

        try:
            # Check pgvector extension installation (cluster-wide, stable info)
            ext_result = await query_raw_with_schema(
                "SELECT extname, extversion, nspname as schema "
                "FROM pg_extension e "
                "JOIN pg_namespace n ON e.extnamespace = n.oid "
                "WHERE extname = 'vector'"
            )
            diagnostics["pgvector_extension"] = ext_result
        except Exception as e:
            diagnostics["pgvector_extension"] = f"Error: {e}"

        logger.error(
            f"Vector type error diagnostics:\n"
            f"  Error: {error}\n"
            f"  search_path: {diagnostics.get('search_path')}\n"
            f"  current_schema: {diagnostics.get('current_schema')}\n"
            f"  user_info: {diagnostics.get('user_info')}\n"
            f"  pgvector_extension: {diagnostics.get('pgvector_extension')}"
        )
    except Exception as diag_error:
        logger.error(f"Failed to collect vector error diagnostics: {diag_error}")
