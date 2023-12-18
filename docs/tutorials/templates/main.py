# Import necessary libraries and modules from the afaas framework and other packages.
import logging
import uuid
from typing import Awaitable, Callable

from AFAAS.interfaces.agent import BaseAgent, Configurable
from AFAAS.core.agents.usercontext.configuration import (
    MyCustomAgentConfiguration,
)  # Import configuration
from AFAAS.core.agents.usercontext.loop import (
    UserContextLoop,
)  # Import the UserContextLoop or your custom loop
from AFAAS.core.agents.usercontext.system import (
    MyCustomAgentSystemSettings,
)  # Import system settings
from AFAAS.core.memory import Memory
from AFAAS.core.planning import SimplePlanner
from AFAAS.core.adapters.openai import AFAASChatOpenAI
from AFAAS.interfaces.workspace import AbstractFileWorkspace


# Define your custom agent class. The class name should reflect its purpose.
class MyCustomAgent(BaseAgent, Configurable):
    """A custom agent class for handling the initialization and management of your agent.
    Inherits from BaseAgent to access essential methods and attributes for agent management.
    """

    # Define default settings and configurations
    CLASS_SYSTEM_SETINGS = MyCustomAgentSystemSettings
    CLASS_CONFIGURATION = MyCustomAgentConfiguration

    def __init__(
        self,
        settings: MyCustomAgentSystemSettings,
        memory: Memory,
        openai_provider: AFAASChatOpenAI,
        workspace: AbstractFileWorkspace,
        planning: SimplePlanner,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
    ):
        """Initialize the agent with associated settings, logger, memory, and other necessary attributes.

        Args:
            settings (MyCustomAgentSystemSettings): The system settings for the agent.
            logger (logging.Logger): The logger instance.
            memory (Memory): The memory instance for storing and retrieving data.
            openai_provider (OpenAIProvider): The provider for interacting with OpenAI.
            workspace (LocalFileWorkspace): The workspace instance for handling the work environment.
            planning (SimplePlanner): The planner instance for managing tasks and objectives.
            user_id (uuid.UUID): The UUID of the user interacting with the agent.
            agent_id (uuid.UUID, optional): The UUID of the agent. Defaults to None.
        """
        super().__init__(
            settings=settings,
            logger=logger,
            memory=memory,
            workspace=workspace,
            user_id=user_id,
            agent_id=agent_id,
        )
        # Specific initializations
        self._openai_provider = openai_provider
        self._planning = planning
        self._loop = MyCustomLoop(agent=self)  # Instantiate your custom loop

    async def start(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        """Start the agent and its associated loop.

        Args:
            user_input_handler (Callable): An async function for handling user input.
            user_message_handler (Callable): An async function for handling user messages.
        """
        return_var = await super().start(
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    async def stop(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        """Stop the agent and its associated loop.

        Args:
            user_input_handler (Callable): An async function for handling user input.
            user_message_handler (Callable): An async function for handling user messages.
        """
        return_var = await super().stop(
            agent=self,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    @classmethod
    def _create_agent_custom_treatment(
        cls, agent_settings: MyCustomAgentSettings, logger: logging.Logger
    ) -> None:
        """Implement any custom treatment necessary for creating agents from settings.

        Args:
            agent_settings (MyCustomAgentSettings): The settings for creating the agent.
            logger (logging.Logger): The logger instance.
        """
        pass

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: MyCustomAgentSettings,
    ) -> dict:
        """Determine the agent name and goals based on the user objective and settings.

        Args:
            user_objective (str): The objective of the user.
            agent_settings (MyCustomAgentSettings): The settings for creating the agent.
            logger (logging.Logger): The logger instance.

        Returns:
            dict: A dictionary containing the agent name and goals.
        """
        # ... implementation details
        pass

    def loop(self) -> MyCustomLoop:
        """Get the loop instance associated with this agent.

        Returns:
            MyCustomLoop: The custom loop instance.
        """
        return self._loop
