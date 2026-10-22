"""Tests for the generic embedding service in search/embeddings.py."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import prisma
import pytest
from prisma import Prisma
from prisma.enums import ContentType

from backend.api.features.search import embeddings, rate_limit


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
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch(
        "backend.api.features.search.embeddings.get_openai_client"
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
    with patch(
        "backend.api.features.search.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = None

        with pytest.raises(
            RuntimeError, match="No embedding-capable LLM client configured"
        ):
            await embeddings.generate_embedding("test text")


@pytest.mark.asyncio(loop_scope="session")
async def test_generate_embedding_api_error():
    """Test embedding generation with API error."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

    with patch(
        "backend.api.features.search.embeddings.get_openai_client"
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

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch(
        "backend.api.features.search.embeddings.get_openai_client"
    ) as mock_get_client:
        mock_get_client.return_value = mock_client

        # Create text that will exceed 8191 tokens
        words = [f"word{i}" for i in range(10000)]
        long_text = " ".join(words)  # ~10000 tokens

        await embeddings.generate_embedding(long_text)

        call_args = mock_client.embeddings.create.call_args
        truncated_text = call_args.kwargs["input"]

        enc = encoding_for_model("text-embedding-3-small")
        actual_tokens = len(enc.encode(truncated_text))

        assert actual_tokens <= 8191
        assert actual_tokens >= 8100


@pytest.mark.asyncio(loop_scope="session")
async def test_get_embedding_stats():
    """Test embedding statistics retrieval."""
    mock_handler = MagicMock()
    mock_handler.get_stats = AsyncMock(
        return_value={
            "total": 100,
            "with_embeddings": 75,
            "without_embeddings": 25,
        }
    )

    with patch(
        "backend.api.features.search.embeddings.CONTENT_HANDLERS",
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
async def test_embedding_to_vector_string():
    """Test embedding to PostgreSQL vector string conversion."""
    embedding = [0.1, 0.2, 0.3, -0.4]
    result = embeddings.embedding_to_vector_string(embedding)
    assert result == "[0.1,0.2,0.3,-0.4]"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_repeated_query_reuses_its_embedding():
    # Unique per run: the cache is the real Redis, shared with later runs.
    query = f"repeat {uuid.uuid4().hex}"
    generate = AsyncMock(return_value=[0.1, 0.2, 0.3])
    with patch.object(embeddings, "generate_embedding", generate):
        first = await embeddings.embed_query(query)
        again = await embeddings.embed_query(f"  {query.replace(' ', '   ')} ")

    assert first == again == [0.1, 0.2, 0.3]
    generate.assert_awaited_once_with(query)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_public_query_past_the_budget_is_not_embedded():
    generate = AsyncMock(return_value=[0.1])
    with (
        patch.object(embeddings, "generate_embedding", generate),
        patch.object(
            embeddings,
            "take_public_search_embedding_slot",
            AsyncMock(return_value=False),
        ),
        pytest.raises(embeddings.PublicSearchBudgetSpent),
    ):
        await embeddings.embed_query(f"spent {uuid.uuid4().hex}", public_search=True)

    generate.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_with_the_budget_off_a_public_query_ignores_a_cached_embedding(
    monkeypatch,
):
    query = f"off {uuid.uuid4().hex}"
    with patch.object(embeddings, "generate_embedding", AsyncMock(return_value=[0.4])):
        await embeddings.embed_query(query)
    monkeypatch.setattr(
        rate_limit.settings.config, "public_search_embeddings_per_minute", 0
    )

    with pytest.raises(embeddings.PublicSearchBudgetSpent):
        await embeddings.embed_query(query, public_search=True)
    assert await embeddings.embed_query(query) == [0.4]


@pytest.mark.asyncio(loop_scope="session")
async def test_only_a_new_public_embedding_takes_from_the_budget():
    query = f"cached {uuid.uuid4().hex}"
    slot = AsyncMock(return_value=False)
    with (
        patch.object(embeddings, "generate_embedding", AsyncMock(return_value=[0.4])),
        patch.object(embeddings, "take_public_search_embedding_slot", slot),
    ):
        await embeddings.embed_query(query)
        assert await embeddings.embed_query(query, public_search=True) == [0.4]

    slot.assert_not_awaited()
