"""Tests for the embedding service.

This module tests:
- create_search_text utility function
- EmbeddingService input validation
- EmbeddingService API interaction (mocked)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.integrations.embeddings import (
    EMBEDDING_DIMENSIONS,
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    EmbeddingService,
    create_search_text,
)


class TestCreateSearchText:
    """Tests for the create_search_text utility function."""

    def test_combines_all_fields(self):
        result = create_search_text("Agent Name", "A cool agent", "Does amazing things")
        assert result == "Agent Name A cool agent Does amazing things"

    def test_handles_empty_name(self):
        result = create_search_text("", "Sub heading", "Description")
        assert result == "Sub heading Description"

    def test_handles_empty_sub_heading(self):
        result = create_search_text("Name", "", "Description")
        assert result == "Name Description"

    def test_handles_empty_description(self):
        result = create_search_text("Name", "Sub heading", "")
        assert result == "Name Sub heading"

    def test_handles_all_empty(self):
        result = create_search_text("", "", "")
        assert result == ""

    def test_handles_none_values(self):
        # The function expects strings but should handle None gracefully
        result = create_search_text(None, None, None)  # type: ignore
        assert result == ""

    def test_preserves_content_strips_outer_whitespace(self):
        # The function joins parts and strips the outer result
        # Internal whitespace in each part is preserved
        result = create_search_text("  Name  ", "  Sub  ", "  Desc  ")
        # Each part is joined with space, then outer strip applied
        assert result == "Name     Sub     Desc"

    def test_handles_only_whitespace(self):
        # Parts that are only whitespace become empty after filter
        result = create_search_text("   ", "   ", "   ")
        assert result == ""


class TestEmbeddingServiceValidation:
    """Tests for EmbeddingService input validation."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with a test API key."""
        with patch("backend.integrations.embeddings.Settings") as mock:
            mock_instance = MagicMock()
            mock_instance.secrets.openai_internal_api_key = "test-api-key"
            mock_instance.secrets.openai_api_key = ""
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def service(self, mock_settings):
        """Create an EmbeddingService instance with mocked settings."""
        with patch("backend.integrations.embeddings.openai.AsyncOpenAI"):
            return EmbeddingService()

    def test_init_requires_api_key(self):
        """Test that initialization fails without an API key."""
        with patch("backend.integrations.embeddings.Settings") as mock:
            mock_instance = MagicMock()
            mock_instance.secrets.openai_internal_api_key = ""
            mock_instance.secrets.openai_api_key = ""
            mock.return_value = mock_instance

            with pytest.raises(ValueError, match="OpenAI API key not configured"):
                EmbeddingService()

    def test_init_accepts_explicit_api_key(self):
        """Test that explicit API key overrides settings."""
        with patch("backend.integrations.embeddings.Settings") as mock:
            mock_instance = MagicMock()
            mock_instance.secrets.openai_internal_api_key = ""
            mock_instance.secrets.openai_api_key = ""
            mock.return_value = mock_instance

            with patch("backend.integrations.embeddings.openai.AsyncOpenAI"):
                service = EmbeddingService(api_key="explicit-key")
                assert service.api_key == "explicit-key"

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, service):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embedding("")

    @pytest.mark.asyncio
    async def test_generate_embedding_whitespace_only(self, service):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embedding("   ")

    @pytest.mark.asyncio
    async def test_generate_embedding_exceeds_max_length(self, service):
        """Test that text exceeding max length raises ValueError."""
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            await service.generate_embedding(long_text)

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, service):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            await service.generate_embeddings([])

    @pytest.mark.asyncio
    async def test_generate_embeddings_exceeds_batch_size(self, service):
        """Test that batch exceeding max size raises ValueError."""
        texts = ["text"] * (MAX_BATCH_SIZE + 1)
        with pytest.raises(ValueError, match="Batch size exceeds maximum"):
            await service.generate_embeddings(texts)

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text_in_batch(self, service):
        """Test that empty text in batch raises ValueError with index."""
        with pytest.raises(ValueError, match="Text at index 1 cannot be empty"):
            await service.generate_embeddings(["valid", "", "also valid"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_long_text_in_batch(self, service):
        """Test that long text in batch raises ValueError with index."""
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="Text at index 2 exceeds maximum length"):
            await service.generate_embeddings(["short", "also short", long_text])


class TestEmbeddingServiceAPI:
    """Tests for EmbeddingService API interaction."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_client.embeddings = MagicMock()
        return mock_client

    @pytest.fixture
    def service_with_mock_client(self, mock_openai_client):
        """Create an EmbeddingService with a mocked OpenAI client."""
        with patch("backend.integrations.embeddings.Settings") as mock_settings:
            mock_instance = MagicMock()
            mock_instance.secrets.openai_internal_api_key = "test-key"
            mock_instance.secrets.openai_api_key = ""
            mock_settings.return_value = mock_instance

            with patch(
                "backend.integrations.embeddings.openai.AsyncOpenAI"
            ) as mock_openai:
                mock_openai.return_value = mock_openai_client
                service = EmbeddingService()
                return service, mock_openai_client

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, service_with_mock_client):
        """Test successful embedding generation."""
        service, mock_client = service_with_mock_client

        # Create mock response
        mock_embedding = [0.1] * EMBEDDING_DIMENSIONS
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await service.generate_embedding("test text")

        assert result == mock_embedding
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, service_with_mock_client):
        """Test successful batch embedding generation."""
        service, mock_client = service_with_mock_client

        # Create mock response with multiple embeddings
        mock_embeddings = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=mock_embeddings[0], index=0),
            MagicMock(embedding=mock_embeddings[1], index=1),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await service.generate_embeddings(["text1", "text2"])

        assert result == mock_embeddings
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_preserves_order(self, service_with_mock_client):
        """Test that batch embeddings are returned in correct order even if API returns out of order."""
        service, mock_client = service_with_mock_client

        # Create mock response with embeddings out of order
        mock_embeddings = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
        mock_response = MagicMock()
        # Return in reverse order
        mock_response.data = [
            MagicMock(embedding=mock_embeddings[1], index=1),
            MagicMock(embedding=mock_embeddings[0], index=0),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await service.generate_embeddings(["text1", "text2"])

        # Should be sorted by index
        assert result[0] == mock_embeddings[0]
        assert result[1] == mock_embeddings[1]
