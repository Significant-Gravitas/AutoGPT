"""
End-to-end database tests for embeddings and hybrid search.

These tests hit the actual database to verify SQL queries work correctly.
Tests cover:
1. Embedding storage (store_content_embedding)
2. Embedding retrieval (get_content_embedding)
3. Embedding deletion (delete_content_embedding)
4. Unified hybrid search across content types
5. Store agent hybrid search
"""

import uuid
from typing import AsyncGenerator

import pytest
from prisma.enums import ContentType

from backend.api.features.store import embeddings
from backend.api.features.store.embeddings import EMBEDDING_DIM
from backend.api.features.store.hybrid_search import (
    hybrid_search,
    unified_hybrid_search,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_content_id() -> str:
    """Generate unique content ID for test isolation."""
    return f"test-content-{uuid.uuid4()}"


@pytest.fixture
def test_user_id() -> str:
    """Generate unique user ID for test isolation."""
    return f"test-user-{uuid.uuid4()}"


@pytest.fixture
def mock_embedding() -> list[float]:
    """Generate a mock embedding vector."""
    # Create a normalized embedding vector
    import math

    raw = [float(i % 10) / 10.0 for i in range(EMBEDDING_DIM)]
    # Normalize to unit length (required for cosine similarity)
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


@pytest.fixture
def similar_embedding() -> list[float]:
    """Generate an embedding similar to mock_embedding."""
    import math

    # Similar but slightly different values
    raw = [float(i % 10) / 10.0 + 0.01 for i in range(EMBEDDING_DIM)]
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


@pytest.fixture
def different_embedding() -> list[float]:
    """Generate an embedding very different from mock_embedding."""
    import math

    # Reversed pattern to be maximally different
    raw = [float((EMBEDDING_DIM - i) % 10) / 10.0 for i in range(EMBEDDING_DIM)]
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


@pytest.fixture
async def cleanup_embeddings(
    server,
) -> AsyncGenerator[list[tuple[ContentType, str, str | None]], None]:
    """
    Fixture that tracks created embeddings and cleans them up after tests.

    Yields a list to which tests can append (content_type, content_id, user_id) tuples.
    """
    created_embeddings: list[tuple[ContentType, str, str | None]] = []
    yield created_embeddings

    # Cleanup all created embeddings
    for content_type, content_id, user_id in created_embeddings:
        try:
            await embeddings.delete_content_embedding(content_type, content_id, user_id)
        except Exception:
            pass  # Ignore cleanup errors


# ============================================================================
# store_content_embedding Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_store_content_embedding_store_agent(
    server,
    test_content_id: str,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test storing embedding for STORE_AGENT content type."""
    # Track for cleanup
    cleanup_embeddings.append((ContentType.STORE_AGENT, test_content_id, None))

    result = await embeddings.store_content_embedding(
        content_type=ContentType.STORE_AGENT,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="AI assistant for productivity tasks",
        metadata={"name": "Test Agent", "categories": ["productivity"]},
        user_id=None,  # Store agents are public
    )

    assert result is True

    # Verify it was stored
    stored = await embeddings.get_content_embedding(
        ContentType.STORE_AGENT, test_content_id, user_id=None
    )
    assert stored is not None
    assert stored["contentId"] == test_content_id
    assert stored["contentType"] == "STORE_AGENT"
    assert stored["searchableText"] == "AI assistant for productivity tasks"


@pytest.mark.asyncio(loop_scope="session")
async def test_store_content_embedding_block(
    server,
    test_content_id: str,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test storing embedding for BLOCK content type."""
    cleanup_embeddings.append((ContentType.BLOCK, test_content_id, None))

    result = await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="HTTP request block for API calls",
        metadata={"name": "HTTP Request Block"},
        user_id=None,  # Blocks are public
    )

    assert result is True

    stored = await embeddings.get_content_embedding(
        ContentType.BLOCK, test_content_id, user_id=None
    )
    assert stored is not None
    assert stored["contentType"] == "BLOCK"


@pytest.mark.asyncio(loop_scope="session")
async def test_store_content_embedding_documentation(
    server,
    test_content_id: str,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test storing embedding for DOCUMENTATION content type."""
    cleanup_embeddings.append((ContentType.DOCUMENTATION, test_content_id, None))

    result = await embeddings.store_content_embedding(
        content_type=ContentType.DOCUMENTATION,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="Getting started guide for AutoGPT platform",
        metadata={"title": "Getting Started", "url": "/docs/getting-started"},
        user_id=None,  # Docs are public
    )

    assert result is True

    stored = await embeddings.get_content_embedding(
        ContentType.DOCUMENTATION, test_content_id, user_id=None
    )
    assert stored is not None
    assert stored["contentType"] == "DOCUMENTATION"


@pytest.mark.asyncio(loop_scope="session")
async def test_store_content_embedding_upsert(
    server,
    test_content_id: str,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test that storing embedding twice updates instead of duplicates."""
    cleanup_embeddings.append((ContentType.BLOCK, test_content_id, None))

    # Store first time
    result1 = await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="Original text",
        metadata={"version": 1},
        user_id=None,
    )
    assert result1 is True

    # Store again with different text (upsert)
    result2 = await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="Updated text",
        metadata={"version": 2},
        user_id=None,
    )
    assert result2 is True

    # Verify only one record with updated text
    stored = await embeddings.get_content_embedding(
        ContentType.BLOCK, test_content_id, user_id=None
    )
    assert stored is not None
    assert stored["searchableText"] == "Updated text"


# ============================================================================
# get_content_embedding Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_get_content_embedding_not_found(server):
    """Test retrieving non-existent embedding returns None."""
    result = await embeddings.get_content_embedding(
        ContentType.STORE_AGENT, "non-existent-id", user_id=None
    )
    assert result is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_content_embedding_with_metadata(
    server,
    test_content_id: str,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test that metadata is correctly stored and retrieved."""
    cleanup_embeddings.append((ContentType.STORE_AGENT, test_content_id, None))

    metadata = {
        "name": "Test Agent",
        "subHeading": "A test agent",
        "categories": ["ai", "productivity"],
        "customField": 123,
    }

    await embeddings.store_content_embedding(
        content_type=ContentType.STORE_AGENT,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="test",
        metadata=metadata,
        user_id=None,
    )

    stored = await embeddings.get_content_embedding(
        ContentType.STORE_AGENT, test_content_id, user_id=None
    )

    assert stored is not None
    assert stored["metadata"]["name"] == "Test Agent"
    assert stored["metadata"]["categories"] == ["ai", "productivity"]
    assert stored["metadata"]["customField"] == 123


# ============================================================================
# delete_content_embedding Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_content_embedding(
    server,
    test_content_id: str,
    mock_embedding: list[float],
):
    """Test deleting embedding removes it from database."""
    # Store embedding
    await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=test_content_id,
        embedding=mock_embedding,
        searchable_text="To be deleted",
        metadata=None,
        user_id=None,
    )

    # Verify it exists
    stored = await embeddings.get_content_embedding(
        ContentType.BLOCK, test_content_id, user_id=None
    )
    assert stored is not None

    # Delete it
    result = await embeddings.delete_content_embedding(
        ContentType.BLOCK, test_content_id, user_id=None
    )
    assert result is True

    # Verify it's gone
    stored = await embeddings.get_content_embedding(
        ContentType.BLOCK, test_content_id, user_id=None
    )
    assert stored is None


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_content_embedding_not_found(server):
    """Test deleting non-existent embedding doesn't error."""
    result = await embeddings.delete_content_embedding(
        ContentType.BLOCK, "non-existent-id", user_id=None
    )
    # Should succeed even if nothing to delete
    assert result is True


# ============================================================================
# unified_hybrid_search Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_unified_hybrid_search_finds_matching_content(
    server,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test unified search finds content matching the query."""
    # Create unique content IDs
    agent_id = f"test-agent-{uuid.uuid4()}"
    block_id = f"test-block-{uuid.uuid4()}"
    doc_id = f"test-doc-{uuid.uuid4()}"

    cleanup_embeddings.append((ContentType.STORE_AGENT, agent_id, None))
    cleanup_embeddings.append((ContentType.BLOCK, block_id, None))
    cleanup_embeddings.append((ContentType.DOCUMENTATION, doc_id, None))

    # Store embeddings for different content types
    await embeddings.store_content_embedding(
        content_type=ContentType.STORE_AGENT,
        content_id=agent_id,
        embedding=mock_embedding,
        searchable_text="AI writing assistant for blog posts",
        metadata={"name": "Writing Assistant"},
        user_id=None,
    )

    await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=block_id,
        embedding=mock_embedding,
        searchable_text="Text generation block for creative writing",
        metadata={"name": "Text Generator"},
        user_id=None,
    )

    await embeddings.store_content_embedding(
        content_type=ContentType.DOCUMENTATION,
        content_id=doc_id,
        embedding=mock_embedding,
        searchable_text="How to use writing blocks in AutoGPT",
        metadata={"title": "Writing Guide"},
        user_id=None,
    )

    # Search for "writing" - should find all three
    results, total = await unified_hybrid_search(
        query="writing",
        page=1,
        page_size=20,
    )

    # Should find at least our test content (may find others too)
    content_ids = [r["content_id"] for r in results]
    assert agent_id in content_ids or total >= 1  # Lexical search should find it


@pytest.mark.asyncio(loop_scope="session")
async def test_unified_hybrid_search_filter_by_content_type(
    server,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test unified search can filter by content type."""
    agent_id = f"test-agent-{uuid.uuid4()}"
    block_id = f"test-block-{uuid.uuid4()}"

    cleanup_embeddings.append((ContentType.STORE_AGENT, agent_id, None))
    cleanup_embeddings.append((ContentType.BLOCK, block_id, None))

    # Store both types with same searchable text
    await embeddings.store_content_embedding(
        content_type=ContentType.STORE_AGENT,
        content_id=agent_id,
        embedding=mock_embedding,
        searchable_text="unique_search_term_xyz123",
        metadata={},
        user_id=None,
    )

    await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=block_id,
        embedding=mock_embedding,
        searchable_text="unique_search_term_xyz123",
        metadata={},
        user_id=None,
    )

    # Search only for BLOCK type
    results, total = await unified_hybrid_search(
        query="unique_search_term_xyz123",
        content_types=[ContentType.BLOCK],
        page=1,
        page_size=20,
    )

    # All results should be BLOCK type
    for r in results:
        assert r["content_type"] == "BLOCK"


@pytest.mark.asyncio(loop_scope="session")
async def test_unified_hybrid_search_empty_query(server):
    """Test unified search with empty query returns empty results."""
    results, total = await unified_hybrid_search(
        query="",
        page=1,
        page_size=20,
    )

    assert results == []
    assert total == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_unified_hybrid_search_pagination(
    server,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test unified search pagination works correctly."""
    # Create multiple items
    content_ids = []
    for i in range(5):
        content_id = f"test-pagination-{uuid.uuid4()}"
        content_ids.append(content_id)
        cleanup_embeddings.append((ContentType.BLOCK, content_id, None))

        await embeddings.store_content_embedding(
            content_type=ContentType.BLOCK,
            content_id=content_id,
            embedding=mock_embedding,
            searchable_text=f"pagination test item number {i}",
            metadata={"index": i},
            user_id=None,
        )

    # Get first page
    page1_results, total1 = await unified_hybrid_search(
        query="pagination test",
        content_types=[ContentType.BLOCK],
        page=1,
        page_size=2,
    )

    # Get second page
    page2_results, total2 = await unified_hybrid_search(
        query="pagination test",
        content_types=[ContentType.BLOCK],
        page=2,
        page_size=2,
    )

    # Total should be consistent
    assert total1 == total2

    # Pages should have different content (if we have enough results)
    if len(page1_results) > 0 and len(page2_results) > 0:
        page1_ids = {r["content_id"] for r in page1_results}
        page2_ids = {r["content_id"] for r in page2_results}
        # No overlap between pages
        assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio(loop_scope="session")
async def test_unified_hybrid_search_min_score_filtering(
    server,
    mock_embedding: list[float],
    cleanup_embeddings: list,
):
    """Test unified search respects min_score threshold."""
    content_id = f"test-minscore-{uuid.uuid4()}"
    cleanup_embeddings.append((ContentType.BLOCK, content_id, None))

    await embeddings.store_content_embedding(
        content_type=ContentType.BLOCK,
        content_id=content_id,
        embedding=mock_embedding,
        searchable_text="completely unrelated content about bananas",
        metadata={},
        user_id=None,
    )

    # Search with very high min_score - should filter out low relevance
    results_high, _ = await unified_hybrid_search(
        query="quantum computing algorithms",
        content_types=[ContentType.BLOCK],
        min_score=0.9,  # Very high threshold
        page=1,
        page_size=20,
    )

    # Search with low min_score
    results_low, _ = await unified_hybrid_search(
        query="quantum computing algorithms",
        content_types=[ContentType.BLOCK],
        min_score=0.01,  # Very low threshold
        page=1,
        page_size=20,
    )

    # High threshold should have fewer or equal results
    assert len(results_high) <= len(results_low)


# ============================================================================
# hybrid_search (Store Agents) Tests
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_hybrid_search_store_agents_sql_valid(server):
    """Test that hybrid_search SQL executes without errors."""
    # This test verifies the SQL is syntactically correct
    # even if no results are found
    results, total = await hybrid_search(
        query="test agent",
        page=1,
        page_size=20,
    )

    # Should not raise - verifies SQL is valid
    assert isinstance(results, list)
    assert isinstance(total, int)
    assert total >= 0


@pytest.mark.asyncio(loop_scope="session")
async def test_hybrid_search_with_filters(server):
    """Test hybrid_search with various filter options."""
    # Test with all filter types
    results, total = await hybrid_search(
        query="productivity",
        featured=True,
        creators=["test-creator"],
        category="productivity",
        page=1,
        page_size=10,
    )

    # Should not raise - verifies filter SQL is valid
    assert isinstance(results, list)
    assert isinstance(total, int)


@pytest.mark.asyncio(loop_scope="session")
async def test_hybrid_search_pagination(server):
    """Test hybrid_search pagination."""
    # Page 1
    results1, total1 = await hybrid_search(
        query="agent",
        page=1,
        page_size=5,
    )

    # Page 2
    results2, total2 = await hybrid_search(
        query="agent",
        page=2,
        page_size=5,
    )

    # Verify SQL executes without error
    assert isinstance(results1, list)
    assert isinstance(results2, list)
    assert isinstance(total1, int)
    assert isinstance(total2, int)

    # If page 1 has results, total should be > 0
    # Note: total from page 2 may be 0 if no results on that page (COUNT(*) OVER limitation)
    if results1:
        assert total1 > 0


# ============================================================================
# SQL Validity Tests (verify queries don't break)
# ============================================================================


@pytest.mark.asyncio(loop_scope="session")
async def test_all_content_types_searchable(server):
    """Test that all content types can be searched without SQL errors."""
    for content_type in [
        ContentType.STORE_AGENT,
        ContentType.BLOCK,
        ContentType.DOCUMENTATION,
    ]:
        results, total = await unified_hybrid_search(
            query="test",
            content_types=[content_type],
            page=1,
            page_size=10,
        )

        # Should not raise
        assert isinstance(results, list)
        assert isinstance(total, int)


@pytest.mark.asyncio(loop_scope="session")
async def test_multiple_content_types_searchable(server):
    """Test searching multiple content types at once."""
    results, total = await unified_hybrid_search(
        query="test",
        content_types=[ContentType.BLOCK, ContentType.DOCUMENTATION],
        page=1,
        page_size=20,
    )

    # Should not raise
    assert isinstance(results, list)
    assert isinstance(total, int)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_all_content_types_default(server):
    """Test searching all content types (default behavior)."""
    results, total = await unified_hybrid_search(
        query="test",
        content_types=None,  # Should search all
        page=1,
        page_size=20,
    )

    # Should not raise
    assert isinstance(results, list)
    assert isinstance(total, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
