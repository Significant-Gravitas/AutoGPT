import abc
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.

    """

    name: str
    prompt_token_cost: float
    completion_token_cost: float
    max_tokens: int


@dataclass
class ModelResponse:
    """Standard response struct for a response from an LLM model."""

    model_info: ModelInfo
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0


class ModelType(abc.ABC):
    """A ModelType abstracts implementation logic for a particular kind of model.

    Implementation details of a ModelType should be agnostic to the provider of the
    model (e.g. OpenAI, Anthropic, Google, etc.) and should focus on the
    details of domain logic.

    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def list_models(self) -> dict[str, ModelInfo]:
        """List all available models."""
        ...

    @abc.abstractmethod
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model."""
        ...


class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models.

    Implementation details of a ModelProvider should handle all translation from
    provider-specific details to the generic ModelType interface.

    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...
