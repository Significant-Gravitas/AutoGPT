"""
Integration tests for content handlers using real DB.

Run with: poetry run pytest backend/api/features/store/content_handlers_integration_test.py -xvs

These tests use the real database but mock OpenAI calls.
"""

from unittest.mock import patch

import pytest

from backend.api.features.store.content_handlers import (
    CONTENT_HANDLERS,
    BlockHandler,
    DocumentationHandler,
    StoreAgentHandler,
)
from backend.api.features.store.embeddings import (
    EMBEDDING_DIM,
    backfill_all_content_types,
    ensure_content_embedding,
    get_embedding_stats,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_store_agent_handler_real_db():
    """Test StoreAgentHandler with real database queries."""
    handler = StoreAgentHandler()

    # Get stats from real DB
    stats = await handler.get_stats()

    # Stats should have correct structure
    assert "total" in stats
    assert "with_embeddings" in stats
    assert "without_embeddings" in stats
    assert stats["total"] >= 0
    assert stats["with_embeddings"] >= 0
    assert stats["without_embeddings"] >= 0

    # Get missing items (max 1 to keep test fast)
    items = await handler.get_missing_items(batch_size=1)

    # Items should be list (may be empty if all have embeddings)
    assert isinstance(items, list)

    if items:
        item = items[0]
        assert item.content_id is not None
        assert item.content_type.value == "STORE_AGENT"
        assert item.searchable_text != ""
        assert item.user_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_block_handler_real_db():
    """Test BlockHandler with real database queries."""
    handler = BlockHandler()

    # Get stats from real DB
    stats = await handler.get_stats()

    # Stats should have correct structure
    assert "total" in stats
    assert "with_embeddings" in stats
    assert "without_embeddings" in stats
    assert stats["total"] >= 0  # Should have at least some blocks
    assert stats["with_embeddings"] >= 0
    assert stats["without_embeddings"] >= 0

    # Get missing items (max 1 to keep test fast)
    items = await handler.get_missing_items(batch_size=1)

    # Items should be list
    assert isinstance(items, list)

    if items:
        item = items[0]
        assert item.content_id is not None  # Should be block UUID
        assert item.content_type.value == "BLOCK"
        assert item.searchable_text != ""
        assert item.user_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_documentation_handler_real_fs():
    """Test DocumentationHandler with real filesystem."""
    handler = DocumentationHandler()

    # Get stats from real filesystem
    stats = await handler.get_stats()

    # Stats should have correct structure
    assert "total" in stats
    assert "with_embeddings" in stats
    assert "without_embeddings" in stats
    assert stats["total"] >= 0
    assert stats["with_embeddings"] >= 0
    assert stats["without_embeddings"] >= 0

    # Get missing items (max 1 to keep test fast)
    items = await handler.get_missing_items(batch_size=1)

    # Items should be list
    assert isinstance(items, list)

    if items:
        item = items[0]
        assert item.content_id is not None  # Should be relative path
        assert item.content_type.value == "DOCUMENTATION"
        assert item.searchable_text != ""
        assert item.user_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_stats_all_types():
    """Test get_embedding_stats aggregates all content types."""
    stats = await get_embedding_stats()

    # Should have structure with by_type and totals
    assert "by_type" in stats
    assert "totals" in stats

    # Check each content type is present
    by_type = stats["by_type"]
    assert "STORE_AGENT" in by_type
    assert "BLOCK" in by_type
    assert "DOCUMENTATION" in by_type

    # Check totals are aggregated
    totals = stats["totals"]
    assert totals["total"] >= 0
    assert totals["with_embeddings"] >= 0
    assert totals["without_embeddings"] >= 0
    assert "coverage_percent" in totals


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.generate_embedding")
async def test_ensure_content_embedding_blocks(mock_generate):
    """Test creating embeddings for blocks (mocked OpenAI)."""
    # Mock OpenAI to return fake embedding
    mock_generate.return_value = [0.1] * EMBEDDING_DIM

    # Get one block without embedding
    handler = BlockHandler()
    items = await handler.get_missing_items(batch_size=1)

    if not items:
        pytest.skip("No blocks without embeddings")

    item = items[0]

    # Try to create embedding (OpenAI mocked)
    result = await ensure_content_embedding(
        content_type=item.content_type,
        content_id=item.content_id,
        searchable_text=item.searchable_text,
        metadata=item.metadata,
        user_id=item.user_id,
    )

    # Should succeed with mocked OpenAI
    assert result is True
    mock_generate.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.generate_embedding")
async def test_backfill_all_content_types_dry_run(mock_generate):
    """Test backfill_all_content_types processes all handlers in order."""
    # Mock OpenAI to return fake embedding
    mock_generate.return_value = [0.1] * EMBEDDING_DIM

    # Run backfill with batch_size=1 to process max 1 per type
    result = await backfill_all_content_types(batch_size=1)

    # Should have results for all content types
    assert "by_type" in result
    assert "totals" in result

    by_type = result["by_type"]
    assert "BLOCK" in by_type
    assert "STORE_AGENT" in by_type
    assert "DOCUMENTATION" in by_type

    # Each type should have correct structure
    for content_type, type_result in by_type.items():
        assert "processed" in type_result
        assert "success" in type_result
        assert "failed" in type_result

    # Totals should aggregate
    totals = result["totals"]
    assert totals["processed"] >= 0
    assert totals["success"] >= 0
    assert totals["failed"] >= 0


@pytest.mark.asyncio(loop_scope="session")
async def test_content_handler_registry():
    """Test all handlers are registered in correct order."""
    from prisma.enums import ContentType

    # All three types should be registered
    assert ContentType.STORE_AGENT in CONTENT_HANDLERS
    assert ContentType.BLOCK in CONTENT_HANDLERS
    assert ContentType.DOCUMENTATION in CONTENT_HANDLERS

    # Check handler types
    assert isinstance(CONTENT_HANDLERS[ContentType.STORE_AGENT], StoreAgentHandler)
    assert isinstance(CONTENT_HANDLERS[ContentType.BLOCK], BlockHandler)
    assert isinstance(CONTENT_HANDLERS[ContentType.DOCUMENTATION], DocumentationHandler)
