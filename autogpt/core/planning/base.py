import abc
import enum
from dataclasses import dataclass
from typing import Any, List

from autogpt.core.configuration.base import Configuration
from autogpt.core.workspace.base import Workspace


class ModelRole(enum.StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


@dataclass
class ModelMessage:
    role: ModelRole
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


ModelPrompt = List[ModelMessage]


@dataclass
class PlanningPromptContext:
    progress: Any  # To be defined (maybe here, as this might be a good place for summarization)
    last_command_result: Any  # To be defined in the command interface
    memories: Any  # List[Memory] # To be defined in the memory interface
    user_feedback: Any  # Probably just a raw string


@dataclass
class SelfFeedbackPromptContext:
    # Using existing args here
    reasoning: str
    plan: List[str]
    thoughts: str
    criticism: str


class Planner(abc.ABC):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    configuration_defaults = {"planner": {}}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @staticmethod
    @abc.abstractmethod
    def construct_objective_prompt_from_user_input(
        user_objective: str,
    ) -> ModelPrompt:
        """Construct a prompt to have the Agent define its goals.

        Args:
            user_objective: The user-defined objective for the agent.

        Returns:
            A prompt to have the Agent define its goals based on the user's input.

        """
        ...

    @abc.abstractmethod
    def construct_planning_prompt_from_context(
        self,
        context: PlanningPromptContext,
    ) -> ModelPrompt:
        """Construct a prompt to have the Agent plan its next action.

        Args:
            context: A context object containing information about the agent's
                       progress, result, memories, and feedback.


        Returns:
            A prompt to have the Agent plan its next action based on the provided
            context.

        """
        ...

    @abc.abstractmethod
    def get_self_feedback_prompt(
        self,
        context: SelfFeedbackPromptContext,
    ) -> ModelPrompt:
        """
        Generates a prompt to have the Agent reflect on its proposed next action.

        Args:
            context: A context object containing information about the agent's
                       reasoning, plan, thoughts, and criticism.

        Returns:
            A self-feedback prompt for the language model based on the given context
            and thoughts.

        """
        ...
