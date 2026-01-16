"""
Integration tests for hybrid search with schema handling.

These tests verify that hybrid search works correctly across different database schemas.
"""

from unittest.mock import patch

import pytest
from prisma.enums import ContentType

from backend.api.features.store import embeddings
from backend.api.features.store.hybrid_search import (
    HybridSearchWeights,
    UnifiedSearchWeights,
    hybrid_search,
    unified_hybrid_search,
)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_with_schema_handling():
    """Test that hybrid search correctly handles database schema prefixes."""
    # Test with a mock query to ensure schema handling works
    query = "test agent"

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        # Mock the query result
        mock_query.return_value = [
            {
                "slug": "test/agent",
                "agent_name": "Test Agent",
                "agent_image": "test.png",
                "creator_username": "test",
                "creator_avatar": "avatar.png",
                "sub_heading": "Test sub-heading",
                "description": "Test description",
                "runs": 10,
                "rating": 4.5,
                "categories": ["test"],
                "featured": False,
                "is_available": True,
                "updated_at": "2024-01-01T00:00:00Z",
                "combined_score": 0.8,
                "semantic_score": 0.7,
                "lexical_score": 0.6,
                "category_score": 0.5,
                "recency_score": 0.4,
                "total_count": 1,
            }
        ]

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM  # Mock embedding

            results, total = await hybrid_search(
                query=query,
                page=1,
                page_size=20,
            )

            # Verify the query was called
            assert mock_query.called
            # Verify the SQL template uses schema_prefix placeholder
            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            assert "{schema_prefix}" in sql_template

            # Verify results
            assert len(results) == 1
            assert total == 1
            assert results[0]["slug"] == "test/agent"


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_with_public_schema():
    """Test hybrid search when using public schema (no prefix needed)."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "public"

        with patch(
            "backend.api.features.store.hybrid_search.query_raw_with_schema"
        ) as mock_query:
            mock_query.return_value = []

            with patch(
                "backend.api.features.store.hybrid_search.embed_query"
            ) as mock_embed:
                mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

                results, total = await hybrid_search(
                    query="test",
                    page=1,
                    page_size=20,
                )

                # Verify the mock was set up correctly
                assert mock_schema.return_value == "public"

                # Results should work even with empty results
                assert results == []
                assert total == 0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_with_custom_schema():
    """Test hybrid search when using custom schema (e.g., 'platform')."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "platform"

        with patch(
            "backend.api.features.store.hybrid_search.query_raw_with_schema"
        ) as mock_query:
            mock_query.return_value = []

            with patch(
                "backend.api.features.store.hybrid_search.embed_query"
            ) as mock_embed:
                mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

                results, total = await hybrid_search(
                    query="test",
                    page=1,
                    page_size=20,
                )

                # Verify the mock was set up correctly
                assert mock_schema.return_value == "platform"

                assert results == []
                assert total == 0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_without_embeddings():
    """Test hybrid search gracefully degrades when embeddings are unavailable."""
    # Mock database to return some results
    mock_results = [
        {
            "slug": "test-agent",
            "agent_name": "Test Agent",
            "agent_image": "test.png",
            "creator_username": "creator",
            "creator_avatar": "avatar.png",
            "sub_heading": "Test heading",
            "description": "Test description",
            "runs": 100,
            "rating": 4.5,
            "categories": ["AI"],
            "featured": False,
            "is_available": True,
            "updated_at": "2025-01-01T00:00:00Z",
            "semantic_score": 0.0,  # Zero because no embedding
            "lexical_score": 0.5,
            "category_score": 0.0,
            "recency_score": 0.1,
            "popularity_score": 0.2,
            "combined_score": 0.3,
            "total_count": 1,
        }
    ]

    with patch("backend.api.features.store.hybrid_search.embed_query") as mock_embed:
        with patch(
            "backend.api.features.store.hybrid_search.query_raw_with_schema"
        ) as mock_query:
            # Simulate embedding failure
            mock_embed.return_value = None
            mock_query.return_value = mock_results

            # Should NOT raise - graceful degradation
            results, total = await hybrid_search(
                query="test",
                page=1,
                page_size=20,
            )

            # Verify it returns results even without embeddings
            assert len(results) == 1
            assert results[0]["slug"] == "test-agent"
            assert total == 1


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_with_filters():
    """Test hybrid search with various filters."""
    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        mock_query.return_value = []

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            # Test with featured filter
            results, total = await hybrid_search(
                query="test",
                featured=True,
                creators=["user1", "user2"],
                category="productivity",
                page=1,
                page_size=10,
            )

            # Verify filters were applied in the query
            call_args = mock_query.call_args
            params = call_args[0][1:]  # Skip SQL template

            # Should have query, query_lower, creators array, category
            assert len(params) >= 4


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_weights():
    """Test hybrid search with custom weights."""
    custom_weights = HybridSearchWeights(
        semantic=0.5,
        lexical=0.3,
        category=0.1,
        recency=0.1,
        popularity=0.0,
    )

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        mock_query.return_value = []

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await hybrid_search(
                query="test",
                weights=custom_weights,
                page=1,
                page_size=20,
            )

            # Verify custom weights were used in the query
            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            params = call_args[0][1:]  # Get all parameters passed

            # Check that SQL uses parameterized weights (not f-string interpolation)
            assert "$" in sql_template  # Verify parameterization is used

            # Check that custom weights are in the params
            assert 0.5 in params  # semantic weight
            assert 0.3 in params  # lexical weight
            assert 0.1 in params  # category and recency weights


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_min_score_filtering():
    """Test hybrid search minimum score threshold."""
    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        # Return results with varying scores
        mock_query.return_value = [
            {
                "slug": "high-score/agent",
                "agent_name": "High Score Agent",
                "combined_score": 0.8,
                "total_count": 1,
                # ... other fields
            }
        ]

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            # Test with custom min_score
            results, total = await hybrid_search(
                query="test",
                min_score=0.5,  # High threshold
                page=1,
                page_size=20,
            )

            # Verify min_score was applied in query
            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            params = call_args[0][1:]  # Get all parameters

            # Check that SQL uses parameterized min_score
            assert "combined_score >=" in sql_template
            assert "$" in sql_template  # Verify parameterization

            # Check that custom min_score is in the params
            assert 0.5 in params


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_pagination():
    """Test hybrid search pagination.

    Pagination happens in SQL (LIMIT/OFFSET), then BM25 reranking is applied
    to the paginated results.
    """
    # Create mock results that SQL would return for a page
    mock_results = [
        {
            "slug": f"agent-{i}",
            "agent_name": f"Agent {i}",
            "agent_image": "test.png",
            "creator_username": "test",
            "creator_avatar": "avatar.png",
            "sub_heading": "Test",
            "description": "Test description",
            "runs": 100 - i,
            "rating": 4.5,
            "categories": ["test"],
            "featured": False,
            "is_available": True,
            "updated_at": "2024-01-01T00:00:00Z",
            "searchable_text": f"Agent {i} test description",
            "combined_score": 0.9 - (i * 0.01),
            "semantic_score": 0.7,
            "lexical_score": 0.6,
            "category_score": 0.5,
            "recency_score": 0.4,
            "popularity_score": 0.3,
            "total_count": 25,
        }
        for i in range(10)  # SQL returns page_size results
    ]

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        mock_query.return_value = mock_results

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            # Test page 2 with page_size 10
            results, total = await hybrid_search(
                query="test",
                page=2,
                page_size=10,
            )

            # Verify results returned
            assert len(results) == 10
            assert total == 25  # Total from SQL COUNT(*) OVER()

            # Verify the SQL query uses page_size and offset
            call_args = mock_query.call_args
            params = call_args[0]
            # Last two params are page_size and offset
            page_size_param = params[-2]
            offset_param = params[-1]
            assert page_size_param == 10
            assert offset_param == 10  # (page 2 - 1) * 10


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_error_handling():
    """Test hybrid search error handling."""
    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        # Simulate database error
        mock_query.side_effect = Exception("Database connection error")

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await hybrid_search(
                    query="test",
                    page=1,
                    page_size=20,
                )

            assert "Database connection error" in str(exc_info.value)


# =============================================================================
# Unified Hybrid Search Tests
# =============================================================================


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

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
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

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_query.return_value = mock_results
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await unified_hybrid_search(
                query="test",
                content_types=[ContentType.BLOCK],
                page=1,
                page_size=20,
            )

            # Verify content_types parameter was passed correctly
            call_args = mock_query.call_args
            params = call_args[0][1:]
            # The content types should be in the params as a list
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

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_query.return_value = mock_results
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await unified_hybrid_search(
                query="test",
                user_id="user-123",
                page=1,
                page_size=20,
            )

            # Verify SQL contains user_id filter
            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            params = call_args[0][1:]

            assert 'uce."userId"' in sql_template
            assert "user-123" in params


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

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_query.return_value = []
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await unified_hybrid_search(
                query="test",
                weights=custom_weights,
                page=1,
                page_size=20,
            )

            # Verify custom weights are in parameters
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
            "semantic_score": 0.0,  # Zero because no embedding
            "lexical_score": 0.8,
            "category_score": 0.0,
            "recency_score": 0.2,
            "combined_score": 0.5,
            "total_count": 1,
        },
    ]

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_query.return_value = mock_results
            mock_embed.return_value = None  # Embedding failure

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
    """Test unified search pagination with BM25 reranking.

    Pagination happens in SQL (LIMIT/OFFSET), then BM25 reranking is applied
    to the paginated results.
    """
    # Create mock results that SQL would return for a page
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
        for i in range(15)  # SQL returns page_size results
    ]

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_query.return_value = mock_results
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await unified_hybrid_search(
                query="test",
                page=3,
                page_size=15,
            )

            # Verify results returned
            assert len(results) == 15
            assert total == 50  # Total from SQL COUNT(*) OVER()

            # Verify the SQL query uses page_size and offset
            call_args = mock_query.call_args
            params = call_args[0]
            # Last two params are page_size and offset
            page_size_param = params[-2]
            offset_param = params[-1]
            assert page_size_param == 15
            assert offset_param == 30  # (page 3 - 1) * 15


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_unified_hybrid_search_schema_prefix():
    """Test unified search uses schema_prefix placeholder."""
    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_query.return_value = []
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            await unified_hybrid_search(
                query="test",
                page=1,
                page_size=20,
            )

            call_args = mock_query.call_args
            sql_template = call_args[0][0]

            # Verify schema_prefix placeholder is used for table references
            assert "{schema_prefix}" in sql_template
            assert '"UnifiedContentEmbedding"' in sql_template


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
