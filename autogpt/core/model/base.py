import abc
import enum
from typing import Callable

from pydantic import BaseModel, Field, validator

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.planning.base import ModelPrompt


class ModelService(str, enum.Enum):
    """A ModelService describes what kind of service the model provides."""

    EMBEDDING: str = "embedding"
    LANGUAGE: str = "language"


class ProviderName(str, enum.Enum):
    OPENAI: str = "openai"


class ModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.

    """

    name: str
    service: ModelService
    provider_name: ProviderName
    prompt_token_cost: float = 0.0
    completion_token_cost: float = 0.0


class ModelResponse(BaseModel):
    """Standard response struct for a response from an LLM model."""

    model_info: ModelInfo
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0


class ModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    name: str
    provider_name: ProviderName
    max_tokens: int | None = None
    temperature: float | None = None


class Model(abc.ABC):
    """A ModelType abstracts implementation logic for a particular kind of model.

    Implementation details of a ModelType should be agnostic to the provider of the
    model (e.g. OpenAI, Anthropic, Google, etc.) and should focus on the
    details of domain logic.

    """

    ...


class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models.

    Implementation details of a ModelProvider should handle all translation from
    provider-specific details to the generic ModelType interface.

    """

    ...


####################
# Embedding Models #
####################

Embedding = list[float]


class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    model_service = ModelService.EMBEDDING
    embedding_dimensions: int


class EmbeddingModelResponse(ModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: Embedding = Field(default_factory=list)

    @classmethod
    @validator("completion_tokens_used")
    def _verify_no_completion_tokens_used(cls, v):
        if v > 0:
            raise ValueError("Embeddings should not have completion tokens used.")
        return v


class EmbeddingModel(Model):
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
    async def create_embedding(
        self,
        text: str,
        model_name: str,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        ...


###################
# Language Models #
###################


class LanguageModelInfo(ModelInfo):
    """Struct for language model information."""

    model_service = ModelService.LANGUAGE


class LanguageModelResponse(ModelResponse):
    """Standard response struct for a response from a language model."""

    content: dict = None


class LanguageModel(Model):
    @abc.abstractmethod
    async def determine_agent_objective(
        self,
        objective_prompt: "ModelPrompt",
        **kwargs,
    ) -> LanguageModelResponse:
        """Chat with a language model to determine the agent's objective.

        Args:
            objective_prompt: The prompt to use to determine the agent's objective.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    async def plan_next_action(
        self,
        planning_prompt: "ModelPrompt",
        **kwargs,
    ) -> LanguageModelResponse:
        """Chat with a language model to plan the agent's next action.

        Args:
            planning_prompt: The prompt to use to plan the agent's next action.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    async def get_self_feedback(
        self,
        self_feedback_prompt: "ModelPrompt",
        **kwargs,
    ) -> LanguageModelResponse:
        """Chat with a language model to get feedback on the agent's performance.

        Args:
            self_feedback_prompt: The prompt to use to get feedback on the agent's
                                    performance.

        Returns:
            The response from the language model.

        """
        ...


class LanguageModelProvider(ModelProvider):
    @abc.abstractmethod
    async def create_language_completion(
        self,
        model_prompt: ModelPrompt,
        model_name: str,
        completion_parser: Callable[[str], dict],
        **kwargs,
    ) -> LanguageModelResponse:
        ...
