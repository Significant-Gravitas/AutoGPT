import abc
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from autogpt.core.agent import Agent
    from autogpt.core.budget import BudgetManager
    from autogpt.core.command import Command, CommandRegistry
    from autogpt.core.configuration import Configuration
    from autogpt.core.llm import LanguageModel
    from autogpt.core.logging import Logger
    from autogpt.core.memory import MemoryBackend
    from autogpt.core.messaging import Message, MessageBroker
    from autogpt.core.planning import Planner
    from autogpt.core.workspace import Workspace

class Wrapper(object):
    """Wraps an object (i hope transparently)
    
    Sourced from: https://stackoverflow.com/a/32188347/4975279
    This one class is licensed as (CC BY-SA 3.0)
    """
    def __init__(self, obj):
        self.__dict__['_wrapped_obj'] = obj
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_obj, attr)
    def __setattr__(self, name, value):
        setattr(self._wrapped_obj, name, value)

class PluginManager(abc.ABC):

    plugins_list = {
        "commands": [],
        "budgets": [],
        "memory": None,
        "model": None,
        "message_listeners": [],
        "planner": None,
        "loggers": []
    }

    @abc.abstractmethod
    def __init__(self, workspace: Workspace, configuration: Configuration, logger: Logger) -> None:
        pass

    @abc.abstractmethod
    def gather_plugins():
        """Gathers the various plugins and stores them in plugin manager until they are ready to swap or register"""
        pass

    @abc.abstractmethod
    def pre_concrete_creation():
        """Runs before the concrete classes are attached to an agent

        Used to swap out the concrete classes in the class
        """
        pass

    @abc.abstractmethod
    def update_concrete_classes(command_registry: CommandRegistry, budget_manager: BudgetManager, language_model: LanguageModel, memory_backend: MemoryBackend, message_broker: MessageBroker, planner: Planner):
        """Updates the concrete classes used within the system"""
        pass

    @abc.abstractmethod
    def wrap_concretes_with_hooks()->typing.Tuple[Workspace, Configuration, Logger, CommandRegistry, BudgetManager, LanguageModel, MemoryBackend, MessageBroker, Planner]:
        pass

    @abc.abstractmethod
    def post_concrete_creation():
        """Runs after the concrete classes are created, but before attaching to an agent
        
        Adds the appropriate plugin items into various registries for the agent to use
        """
        pass