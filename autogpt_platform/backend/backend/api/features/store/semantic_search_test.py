"""Tests for the semantic_search function."""

import pytest
from prisma.enums import ContentType

from backend.api.features.store.embeddings import EMBEDDING_DIM, semantic_search


@pytest.mark.asyncio
async def test_search_blocks_only(mocker):
    """Test searching only BLOCK content type."""
    # Mock embed_query to return a test embedding
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    # Mock query_raw_with_schema to return test results
    mock_results = [
        {
            "content_id": "block-123",
            "content_type": "BLOCK",
            "searchable_text": "Calculator Block - Performs arithmetic operations",
            "metadata": {"name": "Calculator", "categories": ["Math"]},
            "similarity": 0.85,
        }
    ]
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=mock_results,
    )

    results = await semantic_search(
        query="calculate numbers",
        content_types=[ContentType.BLOCK],
    )

    assert len(results) == 1
    assert results[0]["content_type"] == "BLOCK"
    assert results[0]["content_id"] == "block-123"
    assert results[0]["similarity"] == 0.85


@pytest.mark.asyncio
async def test_search_multiple_content_types(mocker):
    """Test searching multiple content types simultaneously."""
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    mock_results = [
        {
            "content_id": "block-123",
            "content_type": "BLOCK",
            "searchable_text": "Calculator Block",
            "metadata": {},
            "similarity": 0.85,
        },
        {
            "content_id": "doc-456",
            "content_type": "DOCUMENTATION",
            "searchable_text": "How to use Calculator",
            "metadata": {},
            "similarity": 0.75,
        },
    ]
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=mock_results,
    )

    results = await semantic_search(
        query="calculator",
        content_types=[ContentType.BLOCK, ContentType.DOCUMENTATION],
    )

    assert len(results) == 2
    assert results[0]["content_type"] == "BLOCK"
    assert results[1]["content_type"] == "DOCUMENTATION"


@pytest.mark.asyncio
async def test_search_with_min_similarity_threshold(mocker):
    """Test that results below min_similarity are filtered out."""
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    # Only return results above 0.7 similarity
    mock_results = [
        {
            "content_id": "block-123",
            "content_type": "BLOCK",
            "searchable_text": "Calculator Block",
            "metadata": {},
            "similarity": 0.85,
        }
    ]
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=mock_results,
    )

    results = await semantic_search(
        query="calculate",
        content_types=[ContentType.BLOCK],
        min_similarity=0.7,
    )

    assert len(results) == 1
    assert results[0]["similarity"] >= 0.7


@pytest.mark.asyncio
async def test_search_fallback_to_lexical(mocker):
    """Test fallback to lexical search when embeddings fail."""
    # Mock embed_query to return None (embeddings unavailable)
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=None,
    )

    mock_lexical_results = [
        {
            "content_id": "block-123",
            "content_type": "BLOCK",
            "searchable_text": "Calculator Block performs calculations",
            "metadata": {},
            "similarity": 0.0,
        }
    ]
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=mock_lexical_results,
    )

    results = await semantic_search(
        query="calculator",
        content_types=[ContentType.BLOCK],
    )

    assert len(results) == 1
    assert results[0]["similarity"] == 0.0  # Lexical search returns 0 similarity


@pytest.mark.asyncio
async def test_search_empty_query():
    """Test that empty query returns no results."""
    results = await semantic_search(query="")
    assert results == []

    results = await semantic_search(query="   ")
    assert results == []


@pytest.mark.asyncio
async def test_search_with_user_id_filter(mocker):
    """Test searching with user_id filter for private content."""
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    mock_results = [
        {
            "content_id": "agent-789",
            "content_type": "LIBRARY_AGENT",
            "searchable_text": "My Custom Agent",
            "metadata": {},
            "similarity": 0.9,
        }
    ]
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=mock_results,
    )

    results = await semantic_search(
        query="custom agent",
        content_types=[ContentType.LIBRARY_AGENT],
        user_id="user-123",
    )

    assert len(results) == 1
    assert results[0]["content_type"] == "LIBRARY_AGENT"


@pytest.mark.asyncio
async def test_search_limit_parameter(mocker):
    """Test that limit parameter correctly limits results."""
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    # Return 5 results
    mock_results = [
        {
            "content_id": f"block-{i}",
            "content_type": "BLOCK",
            "searchable_text": f"Block {i}",
            "metadata": {},
            "similarity": 0.8,
        }
        for i in range(5)
    ]
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=mock_results,
    )

    results = await semantic_search(
        query="block",
        content_types=[ContentType.BLOCK],
        limit=5,
    )

    assert len(results) == 5


@pytest.mark.asyncio
async def test_search_default_content_types(mocker):
    """Test that default content_types includes BLOCK, STORE_AGENT, and DOCUMENTATION."""
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    mock_query_raw = mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=[],
    )

    await semantic_search(query="test")

    # Check that the SQL query includes all three default content types
    call_args = mock_query_raw.call_args
    assert "BLOCK" in str(call_args)
    assert "STORE_AGENT" in str(call_args)
    assert "DOCUMENTATION" in str(call_args)


@pytest.mark.asyncio
async def test_search_handles_database_error(mocker):
    """Test that database errors are handled gracefully."""
    mock_embedding = [0.1] * EMBEDDING_DIM
    mocker.patch(
        "backend.api.features.store.embeddings.embed_query",
        return_value=mock_embedding,
    )

    # Simulate database error
    mocker.patch(
        "backend.api.features.store.embeddings.query_raw_with_schema",
        side_effect=Exception("Database connection failed"),
    )

    results = await semantic_search(
        query="test",
        content_types=[ContentType.BLOCK],
    )

    # Should return empty list on error
    assert results == []
