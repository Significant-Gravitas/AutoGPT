import abc

from autogpt.core.resource.model_providers import (
    EmbeddingModelProviderModelResponse,
)


class EmbeddingModelResponse(EmbeddingModelProviderModelResponse):
    """Standard response struct for a response from an embedding model."""


class EmbeddingModel(abc.ABC):
    @abc.abstractmethod
    async def get_embedding(self, text: str) -> EmbeddingModelResponse:
        """Get the embedding for a prompt.

        Args:
            text: The text to embed.

        Returns:
            The response from the embedding model.

        """
        ...