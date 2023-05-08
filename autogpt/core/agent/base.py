from __future__ import annotations

import abc
import logging
from importlib import import_module
from typing import List, Tuple, Type

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

    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls, workpace_path: Workspace, message_broker: MessageBroker
    ) -> Agent:
        pass

    @abc.abstractmethod
    def run(self):
        pass
