"""
Integration tests for embeddings with schema handling.

These tests verify that embeddings operations work correctly across different database schemas.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ContentType

from backend.api.features.store import embeddings
from backend.api.features.store.embeddings import EMBEDDING_DIM

# Schema prefix tests removed - functionality moved to db.raw_with_schema() helper


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_store_content_embedding_with_schema():
    """Test storing embeddings with proper schema handling."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "platform"

        with patch("prisma.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await embeddings.store_content_embedding(
                content_type=ContentType.STORE_AGENT,
                content_id="test-id",
                embedding=[0.1] * EMBEDDING_DIM,
                searchable_text="test text",
                metadata={"test": "data"},
                user_id=None,
            )

            # Verify the query was called
            assert mock_client.execute_raw.called

            # Get the SQL query that was executed
            call_args = mock_client.execute_raw.call_args
            sql_query = call_args[0][0]

            # Verify schema prefix is in the query
            assert '"platform"."UnifiedContentEmbedding"' in sql_query

            # Verify result
            assert result is True


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_get_content_embedding_with_schema():
    """Test retrieving embeddings with proper schema handling."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "platform"

        with patch("prisma.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.query_raw.return_value = [
                {
                    "contentType": "STORE_AGENT",
                    "contentId": "test-id",
                    "userId": None,
                    "embedding": "[0.1, 0.2]",
                    "searchableText": "test",
                    "metadata": {},
                    "createdAt": "2024-01-01",
                    "updatedAt": "2024-01-01",
                }
            ]
            mock_get_client.return_value = mock_client

            result = await embeddings.get_content_embedding(
                ContentType.STORE_AGENT,
                "test-id",
                user_id=None,
            )

            # Verify the query was called
            assert mock_client.query_raw.called

            # Get the SQL query that was executed
            call_args = mock_client.query_raw.call_args
            sql_query = call_args[0][0]

            # Verify schema prefix is in the query
            assert '"platform"."UnifiedContentEmbedding"' in sql_query

            # Verify result
            assert result is not None
            assert result["contentId"] == "test-id"


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_delete_content_embedding_with_schema():
    """Test deleting embeddings with proper schema handling."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "platform"

        with patch("prisma.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await embeddings.delete_content_embedding(
                ContentType.STORE_AGENT,
                "test-id",
            )

            # Verify the query was called
            assert mock_client.execute_raw.called

            # Get the SQL query that was executed
            call_args = mock_client.execute_raw.call_args
            sql_query = call_args[0][0]

            # Verify schema prefix is in the query
            assert '"platform"."UnifiedContentEmbedding"' in sql_query

            # Verify result
            assert result is True


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_get_embedding_stats_with_schema():
    """Test embedding statistics with proper schema handling via content handlers."""
    # Mock handler to return stats
    mock_handler = MagicMock()
    mock_handler.get_stats = AsyncMock(
        return_value={
            "total": 100,
            "with_embeddings": 80,
            "without_embeddings": 20,
        }
    )

    with patch(
        "backend.api.features.store.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        result = await embeddings.get_embedding_stats()

        # Verify handler was called
        mock_handler.get_stats.assert_called_once()

        # Verify new result structure
        assert "by_type" in result
        assert "totals" in result
        assert result["totals"]["total"] == 100
        assert result["totals"]["with_embeddings"] == 80
        assert result["totals"]["without_embeddings"] == 20
        assert result["totals"]["coverage_percent"] == 80.0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_backfill_missing_embeddings_with_schema():
    """Test backfilling embeddings via content handlers."""
    from backend.api.features.store.content_handlers import ContentItem

    # Create mock content item
    mock_item = ContentItem(
        content_id="version-1",
        content_type=ContentType.STORE_AGENT,
        searchable_text="Test Agent Test description",
        metadata={"name": "Test Agent"},
    )

    # Mock handler
    mock_handler = MagicMock()
    mock_handler.get_missing_items = AsyncMock(return_value=[mock_item])

    with patch(
        "backend.api.features.store.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        with patch(
            "backend.api.features.store.embeddings.generate_embedding",
            return_value=[0.1] * EMBEDDING_DIM,
        ):
            with patch(
                "backend.api.features.store.embeddings.store_content_embedding",
                return_value=True,
            ):
                result = await embeddings.backfill_missing_embeddings(batch_size=10)

                # Verify handler was called
                mock_handler.get_missing_items.assert_called_once_with(10)

                # Verify results
                assert result["processed"] == 1
                assert result["success"] == 1
                assert result["failed"] == 0


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_ensure_content_embedding_with_schema():
    """Test ensuring embeddings exist with proper schema handling."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "platform"

        with patch(
            "backend.api.features.store.embeddings.get_content_embedding"
        ) as mock_get:
            # Simulate no existing embedding
            mock_get.return_value = None

            with patch(
                "backend.api.features.store.embeddings.generate_embedding"
            ) as mock_generate:
                mock_generate.return_value = [0.1] * EMBEDDING_DIM

                with patch(
                    "backend.api.features.store.embeddings.store_content_embedding"
                ) as mock_store:
                    mock_store.return_value = True

                    result = await embeddings.ensure_content_embedding(
                        content_type=ContentType.STORE_AGENT,
                        content_id="test-id",
                        searchable_text="test text",
                        metadata={"test": "data"},
                        user_id=None,
                        force=False,
                    )

                    # Verify the flow
                    assert mock_get.called
                    assert mock_generate.called
                    assert mock_store.called
                    assert result is True


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_backward_compatibility_store_embedding():
    """Test backward compatibility wrapper for store_embedding."""
    with patch(
        "backend.api.features.store.embeddings.store_content_embedding"
    ) as mock_store:
        mock_store.return_value = True

        result = await embeddings.store_embedding(
            version_id="test-version-id",
            embedding=[0.1] * EMBEDDING_DIM,
            tx=None,
        )

        # Verify it calls the new function with correct parameters
        assert mock_store.called
        call_args = mock_store.call_args

        assert call_args[1]["content_type"] == ContentType.STORE_AGENT
        assert call_args[1]["content_id"] == "test-version-id"
        assert call_args[1]["user_id"] is None
        assert result is True


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_backward_compatibility_get_embedding():
    """Test backward compatibility wrapper for get_embedding."""
    with patch(
        "backend.api.features.store.embeddings.get_content_embedding"
    ) as mock_get:
        mock_get.return_value = {
            "contentType": "STORE_AGENT",
            "contentId": "test-version-id",
            "embedding": "[0.1, 0.2]",
            "createdAt": "2024-01-01",
            "updatedAt": "2024-01-01",
        }

        result = await embeddings.get_embedding("test-version-id")

        # Verify it calls the new function
        assert mock_get.called

        # Verify it transforms to old format
        assert result is not None
        assert result["storeListingVersionId"] == "test-version-id"
        assert "embedding" in result


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_schema_handling_error_cases():
    """Test error handling in schema-aware operations."""
    with patch("backend.data.db.get_database_schema") as mock_schema:
        mock_schema.return_value = "platform"

        with patch("prisma.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.execute_raw.side_effect = Exception("Database error")
            mock_get_client.return_value = mock_client

            # Should raise exception on error
            with pytest.raises(Exception, match="Database error"):
                await embeddings.store_content_embedding(
                    content_type=ContentType.STORE_AGENT,
                    content_id="test-id",
                    embedding=[0.1] * EMBEDDING_DIM,
                    searchable_text="test",
                    metadata=None,
                    user_id=None,
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
