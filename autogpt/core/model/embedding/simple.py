import logging

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
)
from autogpt.core.model.embedding.base import (
    EmbeddingModel,
    EmbeddingModelProvider,
    EmbeddingModelResponse,
)


class EmbeddingModelConfiguration(SystemConfiguration):
    """Configuration for the embedding model."""


class SimpleEmbeddingModel(EmbeddingModel, Configurable):

    defaults = SystemSettings(
        name="simple_embedding_model",
        description="A simple embedding model.",
        configuration=EmbeddingModelConfiguration(),
    )

    def __init__(
        self,
        configuration: EmbeddingModelConfiguration,
        logger: logging.Logger,
        model_provider: EmbeddingModelProvider,
    ):
        self._configuration = configuration
        self._logger = logger
        self._model_provider = model_provider

    async def get_embedding(self, text: str) -> EmbeddingModelResponse:
        """Get the embedding for a prompt.

        Args:
            text: The text to embed.

        Returns:
            The response from the embedding model.

        """
        return await self._model_provider.create_embedding(
            text,
            model_name="embedding_model",
            embedding_parser=lambda x: x,
        )
