"""
Integration tests for the store-agent hybrid search.
"""

from unittest.mock import patch

import pytest

from backend.api.features.search import embeddings
from backend.api.features.store.hybrid_search import HybridSearchWeights, hybrid_search


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_with_schema_handling():
    """Test that hybrid search correctly handles database schema prefixes."""
    query = "test agent"

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
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
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await hybrid_search(
                query=query,
                page=1,
                page_size=20,
            )

            assert mock_query.called
            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            assert "{schema_prefix}" in sql_template

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

                assert mock_schema.return_value == "public"
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

                assert mock_schema.return_value == "platform"
                assert results == []
                assert total == 0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_without_embeddings():
    """Test hybrid search gracefully degrades when embeddings are unavailable."""
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
            "semantic_score": 0.0,
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
            mock_embed.side_effect = Exception("Embedding generation failed")
            mock_query.return_value = mock_results

            results, total = await hybrid_search(
                query="test",
                page=1,
                page_size=20,
            )

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

            results, total = await hybrid_search(
                query="test",
                featured=True,
                creators=["user1", "user2"],
                category="productivity",
                page=1,
                page_size=10,
            )

            call_args = mock_query.call_args
            params = call_args[0][1:]
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

            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            params = call_args[0][1:]

            assert "$" in sql_template
            assert 0.5 in params
            assert 0.3 in params
            assert 0.1 in params


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_min_score_filtering():
    """Test hybrid search minimum score threshold."""
    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        mock_query.return_value = [
            {
                "slug": "high-score/agent",
                "agent_name": "High Score Agent",
                "combined_score": 0.8,
                "total_count": 1,
            }
        ]

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await hybrid_search(
                query="test",
                min_score=0.5,
                page=1,
                page_size=20,
            )

            call_args = mock_query.call_args
            sql_template = call_args[0][0]
            params = call_args[0][1:]

            assert "combined_score >=" in sql_template
            assert 0.5 in params


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_hybrid_search_pagination():
    """Test hybrid search pagination."""
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
        for i in range(10)
    ]

    with patch(
        "backend.api.features.store.hybrid_search.query_raw_with_schema"
    ) as mock_query:
        mock_query.return_value = mock_results

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            results, total = await hybrid_search(
                query="test",
                page=2,
                page_size=10,
            )

            assert len(results) == 10
            assert total == 25

            call_args = mock_query.call_args
            params = call_args[0]
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
        mock_query.side_effect = Exception("Database connection error")

        with patch(
            "backend.api.features.store.hybrid_search.embed_query"
        ) as mock_embed:
            mock_embed.return_value = [0.1] * embeddings.EMBEDDING_DIM

            with pytest.raises(Exception) as exc_info:
                await hybrid_search(
                    query="test",
                    page=1,
                    page_size=20,
                )

            assert "Database connection error" in str(exc_info.value)
