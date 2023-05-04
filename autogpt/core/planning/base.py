import abc
import typing

if typing.TYPE_CHECKING:
    from autogpt.core.configuration.base import Configuration
    from autogpt.core.logging.base import Logger
    from autogpt.core.plugin.base import PluginManager
    from autogpt.core.workspace.base import Workspace
    from autogpt.config.ai_config import AIConfig


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
    def construct_goals_prompt(
        big_task: str,
    ) -> AIConfig:
        """This method is called upon the creation of an agent to define in more detail its goals"""
        pass

    @abc.abstractmethod
    def get_self_feedback(
        self,
        thoughts: dict,  # Should this be a thought object?
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
