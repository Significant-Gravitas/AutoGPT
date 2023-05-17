import logging

from autogpt.core.configuration import (
    Configurable,
    SystemSettings,
    SystemConfiguration,
    UserConfigurable,
)
from autogpt.core.embedding.base import (
    EmbeddingModel,
    EmbeddingModelResponse,
)
from autogpt.core.resource.model_providers import (
    EmbeddingModelProvider,
    ModelProviderName,
    OpenAIModelName,
)


class EmbeddingModelConfiguration(SystemConfiguration):
    """Configuration for the embedding model"""
    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()


class EmbeddingModelSettings(SystemSettings):
    configuration: EmbeddingModelConfiguration


class SimpleEmbeddingModel(EmbeddingModel, Configurable):
    defaults = EmbeddingModelSettings(
        name="simple_embedding_model",
        description="A simple embedding model.",
        configuration=EmbeddingModelConfiguration(
            model_name=OpenAIModelName.ADA,
            provider_name=ModelProviderName.OPENAI,
        ),
    )

    def __init__(
        self,
        settings: EmbeddingModelSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, EmbeddingModelProvider],
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._model_provider = model_providers[self._configuration.provider_name]

    async def get_embedding(self, text: str) -> EmbeddingModelResponse:
        """Get the embedding for a prompt.

        Args:
            text: The text to embed.

        Returns:
            The response from the embedding model.

        """
        response = await self._model_provider.create_embedding(
            text,
            model_name="embedding_model",
            embedding_parser=lambda x: x,
        )
        return EmbeddingModelResponse.parse_obj(response.dict())
