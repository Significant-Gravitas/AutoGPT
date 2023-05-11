import abc
from dataclasses import dataclass
from typing import Dict


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


class Model(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all available models."""
        ...

    @abc.abstractmethod
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model."""
        ...
