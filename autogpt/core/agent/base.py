import abc
import logging
from typing import List, Tuple

from autogpt.core.budget import BudgetManager
from autogpt.core.command import CommandRegistry
from autogpt.core.configuration import Configuration
from autogpt.core.llm import LanguageModel
from autogpt.core.logging import Logger
from autogpt.core.memory import MemoryBackend
from autogpt.core.messaging import MessageBroker
from autogpt.core.planning import Planner
from autogpt.core.plugin import PluginManager
from autogpt.core.workspace import Workspace


class Agent(abc.ABC):
    def __init__(
        self,
        configuration: Configuration,
        logger: Logger,
        budget_manager: BudgetManager,
        command_registry: CommandRegistry,
        language_model: LanguageModel,
        memory_backend: MemoryBackend,
        message_broker: MessageBroker,
        planner: Planner,
        plugin_manager: PluginManager,
        workspace: Workspace,
    ):
        self.configuration = configuration
        self.logger = logger

        self.budget_manager = budget_manager
        self.command_registry = command_registry
        self.language_model = language_model
        self.memory_backend = memory_backend
        self.message_broker = message_broker
        self.planner = planner
        self.plugin_manager = plugin_manager
        self.workspace = workspace

    @abc.abstractmethod
    def run(self):
        pass


class AgentFactory:

    configuration_defaults = {
        # Which subsystems to use. These must be subclasses of the base classes,
        # but could come from plugins.
        'system': {
            'agent': 'autogpt.core.agent.Agent',
            'budget_manager': 'autogpt.core.budget.BudgetManager',
            'command_registry': 'autogpt.core.command.CommandRegistry',
            'language_model': 'autogpt.core.llm.LanguageModel',
            'memory_backend': 'autogpt.core.memory.MemoryBackend',
            'planner': 'autogpt.core.planning.Planner',
        }
    }

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def compile_configuration(self, user_configuration: dict) -> Tuple[Configuration, List[str]]:
        """Compile the user's configuration with the defaults."""
        agent_system_configuration = self.configuration_defaults['system']
        agent_system_configuration.update(user_configuration.get('system', {}))
        system_classes = self._get_system_classes(agent_system_configuration)
        system_defaults =

        agent_configuration = Configuration()


    def _get_system_classes(self, agent_system_configuration: dict):
        # Parse the configuration to get the system classes
        # Import those classes by some mechanism, return a list of classes
        return [
            BudgetManager,
            CommandRegistry,
            LanguageModel,
            MemoryBackend,
            Planner,
            Workspace,
        ]

    @abc.abstractmethod
    def create_agent(self, *args, **kwargs) -> Agent:
        pass

