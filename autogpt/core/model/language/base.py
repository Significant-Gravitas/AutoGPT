import abc
import typing
from dataclasses import dataclass
from typing import Dict

from autogpt.core.model.base import Model, ModelInfo, ModelResponse

if typing.TYPE_CHECKING:
    from autogpt.core.configuration import Configuration
    from autogpt.core.planning.base import ModelPrompt
    from autogpt.core.workspace import Workspace


@dataclass
class LanguageModelInfo(ModelInfo):
    """Struct for language model information."""


@dataclass
class LanguageModelResponse(ModelResponse):
    """Standard response struct for a response from a language model."""

    content: str = None


class LanguageModel(Model):
    configuration_defaults = {"language_model": {}}

    @abc.abstractmethod
    def __init__(
        self,
        configuration: "Configuration",
        workspace: "Workspace",
    ):
        ...

    @abc.abstractmethod
    def list_models(self) -> Dict[str, LanguageModelInfo]:
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
