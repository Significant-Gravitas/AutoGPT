from unittest.mock import AsyncMock, MagicMock, patch

import prisma
import pytest
from prisma import Prisma
from prisma.enums import ContentType

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
async def test_build_searchable_text():
    """Test searchable text building from listing fields."""
    result = embeddings.build_searchable_text(
        name="AI Assistant",
        description="A helpful AI assistant for productivity",
        sub_heading="Boost your productivity",
        categories=["AI", "Productivity"],
    )

    expected = "AI Assistant Boost your productivity A helpful AI assistant for productivity AI Productivity"
    assert result == expected


@pytest.mark.asyncio(loop_scope="session")
async def test_build_searchable_text_empty_fields():
    """Test searchable text building with empty fields."""
    result = embeddings.build_searchable_text(
        name="", description="Test description", sub_heading="", categories=[]
    )

    assert result == "Test description"


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_embedding_success():
    """Test successful embedding generation."""
    # Mock OpenAI response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

    # Use AsyncMock for async embeddings.create method
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    # Patch at the point of use in embeddings.py
    with patch(
        "backend.api.features.store.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_client

        result = await embeddings.generate_embedding("test text")

        assert result is not None
        assert len(result) == embeddings.EMBEDDING_DIM
        assert result[0] == 0.1

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_embedding_no_api_key():
    """Test embedding generation without API key."""
    # Patch at the point of use in embeddings.py
    with patch(
        "backend.api.features.store.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = None

        with pytest.raises(RuntimeError, match="openai_internal_api_key not set"):
            await embeddings.generate_embedding("test text")


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_embedding_api_error():
    """Test embedding generation with API error."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

    # Patch at the point of use in embeddings.py
    with patch(
        "backend.api.features.store.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_client

        with pytest.raises(Exception, match="API Error"):
            await embeddings.generate_embedding("test text")


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_embedding_text_truncation():
    """Test that long text is properly truncated using tiktoken."""
    from tiktoken import encoding_for_model

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1] * embeddings.EMBEDDING_DIM

    # Use AsyncMock for async embeddings.create method
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    # Patch at the point of use in embeddings.py
    with patch(
        "backend.api.features.store.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_client

        # Create text that will exceed 8191 tokens
        # Use varied characters to ensure token-heavy text: each word is ~1 token
        words = [f"word{i}" for i in range(10000)]
        long_text = " ".join(words)  # ~10000 tokens

        await embeddings.generate_embedding(long_text)

        # Verify text was truncated to 8191 tokens
        call_args = mock_client.embeddings.create.call_args
        truncated_text = call_args.kwargs["input"]

        # Count actual tokens in truncated text
        enc = encoding_for_model("text-embedding-3-small")
        actual_tokens = len(enc.encode(truncated_text))

        # Should be at or just under 8191 tokens
        assert actual_tokens <= 8191
        # Should be close to the limit (not over-truncated)
        assert actual_tokens >= 8100


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
        "backend.api.features.store.embeddings.query_raw_with_schema",
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
        "backend.api.features.store.embeddings.query_raw_with_schema",
        return_value=[],
    ):
        result = await embeddings.get_embedding("test-version-id")

        assert result is None


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.generate_embedding")
@patch("backend.api.features.store.embeddings.store_embedding")
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
async def test_get_embedding_stats():
    """Test embedding statistics retrieval."""
    # Mock handler stats for each content type
    mock_handler = MagicMock()
    mock_handler.get_stats = AsyncMock(
        return_value={
            "total": 100,
            "with_embeddings": 75,
            "without_embeddings": 25,
        }
    )

    # Patch the CONTENT_HANDLERS where it's used (in embeddings module)
    with patch(
        "backend.api.features.store.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        result = await embeddings.get_embedding_stats()

        assert "by_type" in result
        assert "totals" in result
        assert result["totals"]["total"] == 100
        assert result["totals"]["with_embeddings"] == 75
        assert result["totals"]["without_embeddings"] == 25
        assert result["totals"]["coverage_percent"] == 75.0


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.store_content_embedding")
async def test_backfill_missing_embeddings_success(mock_store):
    """Test backfill with successful embedding generation."""
    # Mock ContentItem from handlers
    from backend.api.features.store.content_handlers import ContentItem

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

    # Mock handler to return missing items
    mock_handler = MagicMock()
    mock_handler.get_missing_items = AsyncMock(return_value=mock_items)

    # Mock store_content_embedding to succeed for first, fail for second
    mock_store.side_effect = [True, False]

    with patch(
        "backend.api.features.store.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        with patch(
            "backend.api.features.store.embeddings.generate_embedding",
            return_value=[0.1] * embeddings.EMBEDDING_DIM,
        ):
            result = await embeddings.backfill_missing_embeddings(batch_size=5)

            assert result["processed"] == 2
            assert result["success"] == 1
            assert result["failed"] == 1
            assert mock_store.call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_backfill_missing_embeddings_no_missing():
    """Test backfill when no embeddings are missing."""
    # Mock handler to return no missing items
    mock_handler = MagicMock()
    mock_handler.get_missing_items = AsyncMock(return_value=[])

    with patch(
        "backend.api.features.store.embeddings.CONTENT_HANDLERS",
        {ContentType.STORE_AGENT: mock_handler},
    ):
        result = await embeddings.backfill_missing_embeddings(batch_size=5)

        assert result["processed"] == 0
        assert result["success"] == 0
        assert result["failed"] == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_embedding_to_vector_string():
    """Test embedding to PostgreSQL vector string conversion."""
    embedding = [0.1, 0.2, 0.3, -0.4]
    result = embeddings.embedding_to_vector_string(embedding)
    assert result == "[0.1,0.2,0.3,-0.4]"


@pytest.mark.asyncio(loop_scope="session")
async def test_embed_query():
    """Test embed_query function (alias for generate_embedding)."""
    with patch(
        "backend.api.features.store.embeddings.generate_embedding"
    ) as mock_generate:
        mock_generate.return_value = [0.1, 0.2, 0.3]

        result = await embeddings.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_generate.assert_called_once_with("test query")
