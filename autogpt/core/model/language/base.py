import abc
from dataclasses import dataclass
from typing import Any, Callable

from autogpt.core.model.base import ModelInfo, ModelProvider, ModelResponse, ModelType
from autogpt.core.planning.base import ModelPrompt


@dataclass
class LanguageModelInfo(ModelInfo):
    """Struct for language model information."""


@dataclass
class LanguageModelResponse(ModelResponse):
    """Standard response struct for a response from a language model."""

    content: dict = None


class LanguageModel(ModelType):
    configuration_defaults = {"language_model": {}}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def list_models(self) -> dict[str, LanguageModelInfo]:
        """List all available models."""
        ...

    @abc.abstractmethod
    def get_model_info(self, model_name: str) -> LanguageModelInfo:
        """Get information about a specific model."""
        ...

    @abc.abstractmethod
    def determine_agent_objective(
        self, objective_prompt: "ModelPrompt"
    ) -> LanguageModelResponse:
        """Chat with a language model to determine the agent's objective.

        Args:
            objective_prompt: The prompt to use to determine the agent's objective.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    def plan_next_action(self, planning_prompt: "ModelPrompt") -> LanguageModelResponse:
        """Chat with a language model to plan the agent's next action.

        Args:
            planning_prompt: The prompt to use to plan the agent's next action.

        Returns:
            The response from the language model.

        """
        ...

    @abc.abstractmethod
    def get_self_feedback(
        self, self_feedback_prompt: "ModelPrompt"
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
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    async def create_language_completion(
        self,
        model_prompt: ModelPrompt,
        model_name: str,
        completion_parser: Callable[[str], dict],
        **kwargs,
    ) -> LanguageModelResponse:
        ...
