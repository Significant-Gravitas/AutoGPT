import abc

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.planning import ModelPrompt
from autogpt.core.resource.model_providers import (
    EmbeddingModelProviderModelResponse,
    LanguageModelProviderModelResponse,
    ModelProviderName,
)


class ModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str
    provider_name: ModelProviderName
    temperature: float | None = None


class Model(abc.ABC):
    """A Model abstracts implementation logic for a particular kind of model.

    Implementation details of a Model should be agnostic to the provider of the
    model (e.g. OpenAI, Anthropic, Google, etc.) and should focus on the
    details of domain logic.

    """

    ...


####################
# Embedding Models #
####################


class EmbeddingModelConfiguration(ModelConfiguration):
    """Configuration for the embedding model"""


class EmbeddingModelResponse(EmbeddingModelProviderModelResponse):
    """Standard response struct for a response from an embedding model."""


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


class LanguageModelResponse(LanguageModelProviderModelResponse):
    """Standard response struct for a response from a language model."""

    pass


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
