import abc
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from autogpt.core.model.base import ModelInfo, ModelProvider, ModelResponse, ModelType

Embedding = list[float]


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    embedding_dimensions: int


@dataclass
class EmbeddingModelResponse(ModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: Embedding = field(default_factory=list)

    def __post_init__(self):
        if self.completion_tokens_used:
            raise ValueError("Embeddings should not have completion tokens used.")


class EmbeddingModel(ModelType):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def list_models(self) -> dict[str, EmbeddingModelInfo]:
        """List all available models."""
        ...

    @abc.abstractmethod
    def get_model_info(self, model_name: str) -> EmbeddingModelInfo:
        """Get information about a specific model."""
        ...

    @abc.abstractmethod
    async def get_embedding(self, text: str) -> EmbeddingModelResponse:
        """Get the embedding for a prompt.

        Args:
            text: The text to embed.

        Returns:
            The response from the embedding model.

        """
        ...


class EmbeddingModelProvider(ModelProvider):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    async def create_embedding(
        self,
        text: str,
        model_name: str,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        ...
