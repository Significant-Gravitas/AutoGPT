import abc
import enum
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from autogpt.config.ai_config import (  # Should this be replaced with Configuration?
        AIConfig,
    )
    from autogpt.core.configuration.base import Configuration
    from autogpt.core.logging.base import Logger
    from autogpt.core.plugin.base import PluginManager
    from autogpt.core.workspace.base import Workspace


class LLMModel(enum.StrEnum):
    # This will be defined probably in the LLM interface module
    GPT3 = "gpt3"
    GPT4 = "gpt4"


class Role(enum.StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    message: str


@dataclass
class Context:  # This will be defined probably in the Memory interface module
    progress: str
    result: str
    memories: list(str)
    feedback: str


@dataclass
class Thoughts:
    system: str
    reasoning: str
    plan: list(str)
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
    def get_self_feedback_prompt(
        self,
        context: Context,
        thoughts: Thoughts,
        llm_model: str,  # Should this be a model object?
    ) -> str:
        """
        Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.
        """
        pass
