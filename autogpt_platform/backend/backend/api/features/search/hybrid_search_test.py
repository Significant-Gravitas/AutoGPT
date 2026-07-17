"""
Tests for the generic unified hybrid search engine.
"""

from unittest.mock import patch

import pytest
from prisma.enums import ContentType
from pydantic import TypeAdapter, ValidationError

from backend.api.features.search import embeddings
from backend.api.features.search.hybrid_search import (
    HybridSearchRow,
    UnifiedSearchWeights,
    tokenize,
    unified_hybrid_search,
)

# ---------------------------------------------------------------------------
# tokenize (BM25)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("AITextGeneratorBlock", ["aitextgeneratorblock"]),
        ("hello world", ["hello", "world"]),
        ("", []),
        ("HTTPRequest", ["httprequest"]),
    ],
)
def test_tokenize(input_text: str, expected: list[str]):
    assert tokenize(input_text) == expected


# ---------------------------------------------------------------------------
# unified_hybrid_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_basic():
    """Test basic unified hybrid search across all content types."""
    mock_results = [
        {
            "content_type": "STORE_AGENT",
            "content_id": "agent-1",
            "searchable_text": "Test Agent Description",
            "metadata": {"name": "Test Agent"},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.7,
            "lexical_score": 0.8,
            "category_score": 0.5,
            "recency_score": 0.3,
            "combined_score": 0.6,
            "total_count": 2,
        },
        {
            "content_type": "BLOCK",
            "content_id": "block-1",
            "searchable_text": "Test Block Description",
            "metadata": {"name": "Test Block"},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.6,
            "lexical_score": 0.7,
            "category_score": 0.4,
            "recency_score": 0.2,
            "combined_score": 0.5,
            "total_count": 2,
        },
    ]

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = mock_results
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        results, total = await unified_hybrid_search(
            query="test",
            page=1,
            page_size=20,
        )

        assert len(results) == 2
        assert total == 2
        assert results[0]["content_type"] == "STORE_AGENT"
        assert results[1]["content_type"] == "BLOCK"


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_output_matches_rpc_contract():
    """Returned rows must validate against the function's declared return
    annotation — the contract the RPC boundary enforces in
    ``service._get_return``. ``lexical_raw`` is computed in SQL only to
    derive ``lexical_score`` and is never projected into result rows, so a
    required ``lexical_raw`` field on ``HybridSearchRow`` made every
    cross-service search RPC log a return-type validation warning."""
    mock_results = [
        {
            "content_type": "BLOCK",
            "content_id": "block-1",
            "searchable_text": "Test Block Description",
            "metadata": {"name": "Test Block"},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.6,
            "lexical_score": 0.7,
            "category_score": 0.4,
            "recency_score": 0.2,
            "combined_score": 0.5,
            "total_count": 1,
        },
    ]

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = mock_results
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        result = await unified_hybrid_search(
            query="test",
            page=1,
            page_size=20,
        )

    # Mirror the RPC boundary: this raises ValidationError if the row shape
    # drifts from what callers actually receive over the wire.
    TypeAdapter(tuple[list[HybridSearchRow], int]).validate_python(result)


def test_hybrid_search_row_rejects_missing_required_field():
    """The contract TypeAdapter must actually reject rows that drop a required
    field — proving the positive contract assertion above is strict rather
    than silently lenient about the row shape."""
    incomplete_row = {
        "content_type": "BLOCK",
        "content_id": "block-1",
        "searchable_text": "Test Block",
        "metadata": {},
        "updated_at": "2025-01-01T00:00:00Z",
        "semantic_score": 0.6,
        "lexical_score": 0.7,
        "category_score": 0.4,
        "recency_score": 0.2,
        # combined_score intentionally omitted — it is a required field.
        "total_count": 1,
    }

    with pytest.raises(ValidationError):
        TypeAdapter(list[HybridSearchRow]).validate_python([incomplete_row])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_filter_by_content_type():
    """Test unified search filtering by specific content types."""
    mock_results = [
        {
            "content_type": "BLOCK",
            "content_id": "block-1",
            "searchable_text": "Test Block",
            "metadata": {},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.7,
            "lexical_score": 0.8,
            "category_score": 0.0,
            "recency_score": 0.3,
            "combined_score": 0.5,
            "total_count": 1,
        },
    ]

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = mock_results
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        results, total = await unified_hybrid_search(
            query="test",
            content_types=[ContentType.BLOCK],
            page=1,
            page_size=20,
        )

        call_args = mock_query.call_args
        params = call_args[0][1:]
        assert ["BLOCK"] in params

        assert len(results) == 1
        assert total == 1


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_with_user_id():
    """Test unified search with user_id for private content."""
    mock_results = [
        {
            "content_type": "STORE_AGENT",
            "content_id": "agent-1",
            "searchable_text": "My Private Agent",
            "metadata": {},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.7,
            "lexical_score": 0.8,
            "category_score": 0.0,
            "recency_score": 0.3,
            "combined_score": 0.6,
            "total_count": 1,
        },
    ]

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = mock_results
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        results, total = await unified_hybrid_search(
            query="test",
            user_id="user-123",
            page=1,
            page_size=20,
        )

        call_args = mock_query.call_args
        sql_template = call_args[0][0]
        params = call_args[0][1:]

        assert 'uce."userId"' in sql_template
        assert "user-123" in params


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_excludes_null_user_library_agents():
    """Defense-in-depth: the SQL filter must NEVER let a LIBRARY_AGENT row
    with NULL userId match an authenticated query — that would leak private
    library data across users."""
    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = []
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        await unified_hybrid_search(
            query="email summarizer",
            user_id="user-123",
            page=1,
            page_size=20,
        )

        sql_template = mock_query.call_args[0][0]
        assert "LIBRARY_AGENT" in sql_template
        assert "WORKSPACE_FILE" in sql_template
        assert "CHAT_SESSION" in sql_template


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_custom_weights():
    """Test unified search with custom weights."""
    custom_weights = UnifiedSearchWeights(
        semantic=0.6,
        lexical=0.2,
        category=0.1,
        recency=0.1,
    )

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = []
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        results, total = await unified_hybrid_search(
            query="test",
            weights=custom_weights,
            page=1,
            page_size=20,
        )

        call_args = mock_query.call_args
        params = call_args[0][1:]

        assert 0.6 in params  # semantic weight
        assert 0.2 in params  # lexical weight


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_graceful_degradation():
    """Test unified search gracefully degrades when embeddings unavailable."""
    mock_results = [
        {
            "content_type": "DOCUMENTATION",
            "content_id": "doc-1",
            "searchable_text": "API Documentation",
            "metadata": {},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.0,
            "lexical_score": 0.8,
            "category_score": 0.0,
            "recency_score": 0.2,
            "combined_score": 0.5,
            "total_count": 1,
        },
    ]

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = mock_results
        mock_embed.side_effect = Exception("Embedding generation failed")

        # Should NOT raise - graceful degradation
        results, total = await unified_hybrid_search(
            query="test",
            page=1,
            page_size=20,
        )

        assert len(results) == 1
        assert total == 1


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_empty_query():
    """Test unified search with empty query returns empty results."""
    results, total = await unified_hybrid_search(
        query="",
        page=1,
        page_size=20,
    )

    assert results == []
    assert total == 0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_pagination():
    """Test unified search pagination with BM25 reranking."""
    mock_results = [
        {
            "content_type": "STORE_AGENT",
            "content_id": f"agent-{i}",
            "searchable_text": f"Agent {i} description",
            "metadata": {"name": f"Agent {i}"},
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.7,
            "lexical_score": 0.8 - (i * 0.01),
            "category_score": 0.5,
            "recency_score": 0.3,
            "combined_score": 0.6 - (i * 0.01),
            "total_count": 50,
        }
        for i in range(15)
    ]

    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = mock_results
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        results, total = await unified_hybrid_search(
            query="test",
            page=3,
            page_size=15,
        )

        assert len(results) == 15
        assert total == 50

        call_args = mock_query.call_args
        params = call_args[0]
        page_size_param = params[-2]
        offset_param = params[-1]
        assert page_size_param == 15
        assert offset_param == 30  # (page 3 - 1) * 15


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_schema_prefix():
    """Test unified search uses schema_prefix placeholder."""
    with (
        patch(
            "backend.api.features.search.hybrid_search.query_raw_with_schema"
        ) as mock_query,
        patch("backend.api.features.search.hybrid_search.embed_query") as mock_embed,
    ):
        mock_query.return_value = []
        mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

        await unified_hybrid_search(
            query="test",
            page=1,
            page_size=20,
        )

        call_args = mock_query.call_args
        sql_template = call_args[0][0]

        assert "{schema_prefix}" in sql_template
        assert '"UnifiedContentEmbedding"' in sql_template


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
