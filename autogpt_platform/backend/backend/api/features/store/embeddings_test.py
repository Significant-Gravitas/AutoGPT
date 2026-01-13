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
        assert len(result) == 1536
        assert result[0] == 0.1

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.util.clients.get_openai_client")
async def test_generate_embedding_no_api_key(mock_get_client):
    """Test embedding generation without API key."""
    mock_get_client.return_value = None

    result = await embeddings.generate_embedding("test text")

    assert result is None


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.util.clients.get_openai_client")
async def test_generate_embedding_api_error(mock_get_client):
    """Test embedding generation with API error."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = Exception("API Error")
    mock_get_client.return_value = mock_client

    result = await embeddings.generate_embedding("test text")

    assert result is None


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_embedding_text_truncation():
    """Test that long text is properly truncated."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1] * 1536

    # Use AsyncMock for async embeddings.create method
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    # Patch at the point of use in embeddings.py
    with patch(
        "backend.api.features.store.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_client

        # Create text longer than 32k chars
        long_text = "a" * 35000

        await embeddings.generate_embedding(long_text)

        # Verify truncated text was sent to API
        call_args = mock_client.embeddings.create.call_args
        assert len(call_args.kwargs["input"]) == 32000


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
    mock_client.execute_raw.assert_called_once()
    call_args = mock_client.execute_raw.call_args[0]
    assert "test-version-id" in call_args
    assert "[0.1,0.2,0.3]" in call_args
    assert None in call_args  # userId should be None for store agents


@pytest.mark.asyncio(loop_scope="session")
async def test_store_embedding_database_error(mocker):
    """Test embedding storage with database error."""
    mock_client = mocker.AsyncMock()
    mock_client.execute_raw.side_effect = Exception("Database error")

    embedding = [0.1, 0.2, 0.3]

    result = await embeddings.store_embedding(
        version_id="test-version-id", embedding=embedding, tx=mock_client
    )

    assert result is False


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_success(mocker):
    """Test successful embedding retrieval."""
    mock_client = mocker.AsyncMock()
    mock_result = [
        {
            "contentType": "STORE_AGENT",
            "contentId": "test-version-id",
            "embedding": "[0.1,0.2,0.3]",
            "searchableText": "Test text",
            "metadata": {},
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z",
        }
    ]
    mock_client.query_raw.return_value = mock_result

    with patch("prisma.get_client", return_value=mock_client):
        result = await embeddings.get_embedding("test-version-id")

        assert result is not None
        assert result["storeListingVersionId"] == "test-version-id"
        assert result["embedding"] == "[0.1,0.2,0.3]"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_not_found(mocker):
    """Test embedding retrieval when not found."""
    mock_client = mocker.AsyncMock()
    mock_client.query_raw.return_value = []

    with patch("prisma.get_client", return_value=mock_client):
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
    mock_generate.return_value = None

    result = await embeddings.ensure_embedding(
        version_id="test-id",
        name="Test",
        description="Test description",
        sub_heading="Test heading",
        categories=["test"],
    )

    assert result is False


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_stats(mocker):
    """Test embedding statistics retrieval."""
    mock_client = mocker.AsyncMock()

    # Mock approved count query
    mock_approved_result = [{"count": 100}]
    # Mock embedded count query
    mock_embedded_result = [{"count": 75}]

    mock_client.query_raw.side_effect = [mock_approved_result, mock_embedded_result]

    with patch("prisma.get_client", return_value=mock_client):
        result = await embeddings.get_embedding_stats()

        assert result["total_approved"] == 100
        assert result["with_embeddings"] == 75
        assert result["without_embeddings"] == 25
        assert result["coverage_percent"] == 75.0


@pytest.mark.asyncio(loop_scope="session")
@patch("backend.api.features.store.embeddings.ensure_embedding")
async def test_backfill_missing_embeddings_success(mock_ensure, mocker):
    """Test backfill with successful embedding generation."""
    mock_client = mocker.AsyncMock()

    # Mock missing embeddings query
    mock_missing = [
        {
            "id": "version-1",
            "name": "Agent 1",
            "description": "Description 1",
            "subHeading": "Heading 1",
            "categories": ["AI"],
        },
        {
            "id": "version-2",
            "name": "Agent 2",
            "description": "Description 2",
            "subHeading": "Heading 2",
            "categories": ["Productivity"],
        },
    ]
    mock_client.query_raw.return_value = mock_missing

    # Mock ensure_embedding to succeed for first, fail for second
    mock_ensure.side_effect = [True, False]

    with patch("prisma.get_client", return_value=mock_client):
        result = await embeddings.backfill_missing_embeddings(batch_size=5)

        assert result["processed"] == 2
        assert result["success"] == 1
        assert result["failed"] == 1
        assert mock_ensure.call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_backfill_missing_embeddings_no_missing(mocker):
    """Test backfill when no embeddings are missing."""
    mock_client = mocker.AsyncMock()
    mock_client.query_raw.return_value = []

    with patch("prisma.get_client", return_value=mock_client):
        result = await embeddings.backfill_missing_embeddings(batch_size=5)

        assert result["processed"] == 0
        assert result["success"] == 0
        assert result["failed"] == 0
        assert result["message"] == "No missing embeddings"


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
