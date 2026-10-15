"""Tests for store-listing embedding wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import prisma
import pytest
from prisma import Prisma
from prisma.enums import ContentType

from backend.api.features.search import embeddings as search_embeddings
from backend.api.features.store import embeddings


@pytest.fixture(autouse=True)
async def setup_prisma():
    """Setup Prisma client for tests."""
    try:
        Prisma()
    except prisma.errors.ClientAlreadyRegisteredError:
        pass
    yield


@pytest.mark.asyncio(loop_scope="session")
async def test_store_embedding_success(mocker):
    """Test successful embedding storage."""
    mock_client = mocker.AsyncMock()
    mock_client.execute_raw = mocker.AsyncMock()

    embedding = [0.1, 0.2, 0.3]

    result = await embeddings.store_embedding(
        version_id="test-version-id", embedding=embedding, tx=mock_client
    )

    assert result is True
    # execute_raw is called once for INSERT (no separate SET search_path needed)
    assert mock_client.execute_raw.call_count == 1

    # Verify the INSERT query with the actual data
    call_args = mock_client.execute_raw.call_args_list[0][0]
    assert "test-version-id" in call_args
    assert "[0.1,0.2,0.3]" in call_args
    assert None in call_args  # userId should be None for store agents


@pytest.mark.asyncio(loop_scope="session")
async def test_store_embedding_database_error(mocker):
    """Test embedding storage with database error."""
    mock_client = mocker.AsyncMock()
    mock_client.execute_raw.side_effect = Exception("Database error")

    embedding = [0.1, 0.2, 0.3]

    with pytest.raises(Exception, match="Database error"):
        await embeddings.store_embedding(
            version_id="test-version-id", embedding=embedding, tx=mock_client
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_success():
    """Test successful embedding retrieval."""
    mock_result = [
        {
            "contentType": "STORE_AGENT",
            "contentId": "test-version-id",
            "userId": None,
            "embedding": "[0.1,0.2,0.3]",
            "searchableText": "Test text",
            "metadata": {},
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
        }
    ]

    with patch(
        "backend.api.features.search.embeddings.query_raw_with_schema",
        return_value=mock_result,
    ):
        result = await embeddings.get_embedding("test-version-id")

        assert result is not None
        assert result["storeListingVersionId"] == "test-version-id"
        assert result["embedding"] == "[0.1,0.2,0.3]"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_not_found():
    """Test embedding retrieval when not found."""
    with patch(
        "backend.api.features.search.embeddings.query_raw_with_schema",
        return_value=[],
    ):
        result = await embeddings.get_embedding("test-version-id")

        assert result is None


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.generate_embedding")
@patch("backend.api.features.store.embeddings.store_content_embedding")
@patch("backend.api.features.store.embeddings.get_embedding")
async def test_ensure_embedding_already_exists(mock_get, mock_store, mock_generate):
    """Test ensure_embedding when embedding already exists."""
    mock_get.return_value = {"embedding": "[0.1,0.2,0.3]"}

    result = await embeddings.ensure_embedding(
        version_id="test-id",
        name="Test",
        description="Test description",
        sub_heading="Test heading",
        categories=["test"],
    )

    assert result is True
    mock_generate.assert_not_called()
    mock_store.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.generate_embedding")
@patch("backend.api.features.store.embeddings.store_content_embedding")
@patch("backend.api.features.store.embeddings.get_embedding")
async def test_ensure_embedding_create_new(mock_get, mock_store, mock_generate):
    """Test ensure_embedding creating new embedding."""
    mock_get.return_value = None
    mock_generate.return_value = [0.1, 0.2, 0.3]
    mock_store.return_value = True

    result = await embeddings.ensure_embedding(
        version_id="test-id",
        name="Test",
        description="Test description",
        sub_heading="Test heading",
        categories=["test"],
    )

    assert result is True
    mock_generate.assert_called_once_with("Test Test heading Test description test")
    mock_store.assert_called_once_with(
        content_type=ContentType.STORE_AGENT,
        content_id="test-id",
        embedding=[0.1, 0.2, 0.3],
        searchable_text="Test Test heading Test description test",
        metadata={"name": "Test", "subHeading": "Test heading", "categories": ["test"]},
        user_id=None,
        tx=None,
    )


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.generate_embedding")
@patch("backend.api.features.store.embeddings.get_embedding")
async def test_ensure_embedding_generation_fails(mock_get, mock_generate):
    """Test ensure_embedding when generation fails."""
    mock_get.return_value = None
    mock_generate.side_effect = Exception("Generation failed")

    with pytest.raises(Exception, match="Generation failed"):
        await embeddings.ensure_embedding(
            version_id="test-id",
            name="Test",
            description="Test description",
            sub_heading="Test heading",
            categories=["test"],
        )


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.search.embeddings.store_content_embedding")
async def test_backfill_missing_embeddings_success(mock_store):
    """Test backfill with successful embedding generation."""
    from backend.api.features.search.content_handlers import ContentItem

    mock_items = [
        ContentItem(
            content_id="version-1",
            content_type=ContentType.STORE_AGENT,
            searchable_text="Agent 1 Description 1",
            metadata={"name": "Agent 1"},
        ),
        ContentItem(
            content_id="version-2",
            content_type=ContentType.STORE_AGENT,
            searchable_text="Agent 2 Description 2",
            metadata={"name": "Agent 2"},
        ),
    ]

    mock_handler = MagicMock()
    mock_handler.get_missing_items = AsyncMock(return_value=mock_items)

    mock_store.side_effect = [True, False]

    with patch(
        "backend.api.features.search.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        with patch(
            "backend.api.features.search.embeddings.get_content_embedding",
            AsyncMock(return_value=None),
        ):
            with patch(
                "backend.api.features.search.embeddings.generate_embedding",
                return_value=[0.1] * search_embeddings.EMBEDDING_DIM,
            ):
                result = await embeddings.backfill_missing_embeddings(batch_size=5)

                assert result["processed"] == 2
                assert result["success"] == 1
                assert result["failed"] == 1
                assert mock_store.call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_backfill_missing_embeddings_no_missing():
    """Test backfill when no embeddings are missing."""
    mock_handler = MagicMock()
    mock_handler.get_missing_items = AsyncMock(return_value=[])

    with patch(
        "backend.api.features.search.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        result = await embeddings.backfill_missing_embeddings(batch_size=5)

        assert result["processed"] == 0
        assert result["success"] == 0
        assert result["failed"] == 0
