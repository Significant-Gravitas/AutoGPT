import abc
import typing
from dataclasses import dataclass, field
from typing import Dict, List

from autogpt.core.model.base import Model, ModelInfo, ModelResponse

if typing.TYPE_CHECKING:
    from autogpt.core.configuration import Configuration


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    embedding_dimensions: int


@dataclass
class EmbeddingModelResponse(ModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.completion_tokens_used:
            raise ValueError("Embeddings should not have completion tokens used.")


class EmbeddingModel(Model):
    configuration_defaults = {"embedding_model": {}}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def list_models(self) -> Dict[str, EmbeddingModelInfo]:
        """List all available models."""
        ...

    @abc.abstractmethod
    def get_model_info(self, model_name: str) -> EmbeddingModelInfo:
        """Get information about a specific model."""
        ...

    @abc.abstractmethod
    def get_embedding(self, text: str) -> EmbeddingModelResponse:
        """Get the embedding for a prompt.

        Args:
            text: The text to embed.

        Returns:
            The response from the embedding model.

        """
        ...
