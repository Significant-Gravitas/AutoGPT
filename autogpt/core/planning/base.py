import abc
import enum
import typing
from dataclasses import dataclass
from typing import Any, List

if typing.TYPE_CHECKING:
    from autogpt.core.configuration.base import Configuration
    from autogpt.core.logging.base import Logger
    from autogpt.core.plugin.base import PluginManager
    from autogpt.core.workspace.base import Workspace


class Role(enum.StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    message: str


ModelPrompt = List[Message]


@dataclass
class PlanningPromptContext:
    progress: Any  # To be defined (maybe here, as this might be a good place for summarization)
    last_command_result: Any  # To be defined in the command interface
    memories: Any  # List[Memory] # To be defined in the memory interface
    user_feedback: Any  # Probably just a raw string


@dataclass
class SelfCriticismPromptContext:
    # Using existing args here
    reasoning: str
    plan: List[str]
    thoughts: str
    criticism: str


class Planner(abc.ABC):
    """Build prompts based on inputs, can potentially store and retrieve planning state from the workspace"""

    @abc.abstractmethod
    def __init__(
        self,
        configuration: Configuration,
        logger: Logger,
        plugin_manager: PluginManager,
        workspace: Workspace,
    ) -> None:
        self.configuration = configuration
        self.logger = logger
        self.plugin_manager = plugin_manager
        self.workspace = workspace
        pass

    @abc.abstractmethod
    def construct_objective_prompt_from_user_input(
        self,
        user_objective: str,
    ) -> list[Message]:  #
        """
        This method is called upon the creation of an agent to refine its goals based on user input.

        Args:
            user_objective (str): The user-defined objective for the agent.

        Returns:
            List[Dict]: A list of message dictionaries that define the refined goals for the agent.
        """
        pass

    @abc.abstractmethod
    def construct_planning_prompt_from_context(
        self,
        context: PlanningPromptContext,
    ) -> ModelPrompt:
        ...

    @abc.abstractmethod
    def get_self_feedback_prompt(
        self,
        context: SelfCriticismPromptContext,
    ) -> ModelPrompt:
        """
        Generates a self-feedback prompt for the language model, based on the provided context and thoughts.

        This method takes in a Context object containing information about the agent's progress, result,
        memories, and feedback. It also takes in a Thoughts object containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. The method combines these elements to create a prompt that
        facilitates self-assessment and improvement for the agent.

        Args:
            context (Context): An object containing information about the agent's progress, result,
                               memories, and feedback.

        Returns:
            ModelPrompt: A self-feedback prompt for the language model based on the given context and thoughts.
        """
        pass
