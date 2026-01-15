"""
Integration tests for hybrid search with schema handling.

These tests verify that hybrid search works correctly across different database schemas.
"""

from unittest.mock import patch

import pytest

from backend.api.features.store.hybrid_search import HybridSearchWeights, hybrid_search


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
            mock_embed.return_value = [0.1] * 1536  # Mock embedding

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
                mock_embed.return_value = [0.1] * 1536

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
                mock_embed.return_value = [0.1] * 1536

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
    """Test hybrid search fails fast when embeddings are unavailable."""
    # Patch where the function is used, not where it's defined
    with patch("backend.api.features.store.hybrid_search.embed_query") as mock_embed:
        # Simulate embedding failure
        mock_embed.return_value = None

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            await hybrid_search(
                query="test",
                page=1,
                page_size=20,
            )

        # Verify error message is generic (doesn't leak implementation details)
        assert "Search service temporarily unavailable" in str(exc_info.value)


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
            mock_embed.return_value = [0.1] * 1536

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
            mock_embed.return_value = [0.1] * 1536

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
            mock_embed.return_value = [0.1] * 1536

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
    """Test hybrid search pagination."""
    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        mock_query.return_value = []

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * 1536

            # Test page 2 with page_size 10
            results, total = await hybrid_search(
                query="test",
                page=2,
                page_size=10,
            )

            # Verify pagination parameters
            call_args = mock_query.call_args
            params = call_args[0]

            # Last two params should be LIMIT and OFFSET
            limit = params[-2]
            offset = params[-1]

            assert limit == 10  # page_size
            assert offset == 10  # (page - 1) * page_size = (2 - 1) * 10


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
            mock_embed.return_value = [0.1] * 1536

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                await hybrid_search(
                    query="test",
                    page=1,
                    page_size=20,
                )

            assert "Database connection error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
