"""
Embedding service for generating text embeddings using OpenAI.

Used for vector-based semantic search in the store.
"""

import functools
import logging
from typing import Optional

import openai

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

# Model configuration
# Using text-embedding-3-small (1536 dimensions) for compatibility with pgvector indexes
# pgvector IVFFlat/HNSW indexes have dimension limits (2000 for IVFFlat, varies for HNSW)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Input validation limits
# OpenAI text-embedding-3-large supports up to 8191 tokens (~32k chars)
# We set a conservative limit to prevent abuse
MAX_TEXT_LENGTH = 10000  # characters
MAX_BATCH_SIZE = 100  # maximum texts per batch request


class EmbeddingService:
    """Service for generating text embeddings using OpenAI.

    The service can be created without an API key - the key is validated
    only when the client property is first accessed. This allows the service
    to be instantiated at module load time without requiring configuration.
    """

    def __init__(self, api_key: Optional[str] = None):
        settings = Settings()
        self.api_key = (
            api_key
            or settings.secrets.openai_internal_api_key
            or settings.secrets.openai_api_key
        )

    @functools.cached_property
    def client(self) -> openai.AsyncOpenAI:
        """Lazily create the OpenAI client, raising if no API key is configured."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY or OPENAI_INTERNAL_API_KEY environment variable."
            )
        return openai.AsyncOpenAI(api_key=self.api_key)

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to generate an embedding for.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If the text is empty or exceeds maximum length.
            openai.APIError: If the OpenAI API call fails.
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters"
            )

        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        if not response.data:
            raise ValueError("OpenAI API returned empty embedding data")
        return response.data[0].embedding

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If any text is invalid or batch size exceeds limit.
            openai.APIError: If the OpenAI API call fails.
        """
        # Input validation
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE} texts")
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters"
                )

        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        # Sort by index to ensure correct ordering
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


def create_search_text(name: str, sub_heading: str, description: str) -> str:
    """
    Combine fields into searchable text for embedding.

    This creates a single text string from the agent's name, sub-heading,
    and description, which is then converted to an embedding vector.

    Args:
        name: The agent name.
        sub_heading: The agent sub-heading/tagline.
        description: The agent description.

    Returns:
        A single string combining all non-empty fields.
    """
    parts = [name or "", sub_heading or "", description or ""]
    # filter(None, parts) removes empty strings since empty string is falsy
    return " ".join(filter(None, parts)).strip()


@functools.cache
def get_embedding_service() -> EmbeddingService:
    """
    Get or create the embedding service singleton.

    Uses functools.cache for thread-safe lazy initialization.

    Returns:
        The shared EmbeddingService instance.

    Raises:
        ValueError: If OpenAI API key is not configured (when generating embeddings).
    """
    return EmbeddingService()
