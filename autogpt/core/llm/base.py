import abc
import typing
from dataclasses import dataclass, field
from typing import Dict, List

from autogpt.core.configuration.base import Configuration
from autogpt.core.logging.base import Logger
from autogpt.core.planning.base import ModelPrompt
from autogpt.core.workspace.base import Workspace


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
class ChatModelInfo(ModelInfo):
    """Struct for chat model information."""

    pass


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    embedding_dimensions: int


@dataclass
class LLMResponse:
    """Standard response struct for a response from an LLM model."""

    model_info: ModelInfo
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0


@dataclass
class EmbeddingModelResponse(LLMResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.completion_tokens_used:
            raise ValueError("Embeddings should not have completion tokens used.")


@dataclass
class ChatModelResponse(LLMResponse):
    """Standard response struct for a response from an LLM model."""

    content: str = None


class LanguageModel(abc.ABC):
    configuration_defaults = {"language_model": {}}

    @abc.abstractmethod
    def __init__(
        self,
        configuration: Configuration,
        logger: Logger,
        workspace: Workspace,
    ):
        ...

    @abc.abstractmethod
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all available models."""
        ...

    @abc.abstractmethod
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a specific model."""
        ...

    @abc.abstractmethod
    def determine_agent_objective(
        self, objective_prompt: ModelPrompt
    ) -> ChatModelResponse:
        """Chat with a language model to determine the agent's objective.

        Args:
            objective_prompt: The prompt to use to determine the agent's objective.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    def plan_next_action(self, planning_prompt: ModelPrompt) -> ChatModelResponse:
        """Chat with a language model to plan the agent's next action.

        Args:
            planning_prompt: The prompt to use to plan the agent's next action.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    def get_self_feedback(self, self_feedback_prompt: ModelPrompt) -> ChatModelResponse:
        """Chat with a language model to get feedback on the agent's performance.

        Args:
            self_feedback_prompt: The prompt to use to get feedback on the agent's
                                    performance.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    def get_embedding(self, data: str) -> EmbeddingModelResponse:
        """Get an embedding for a piece of data.

        Args:
            data: The data to get an embedding for.

        Returns:
            The response from the language model.

        """
        ...
