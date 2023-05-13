from __future__ import annotations

import logging
from typing import List, Tuple, Type

from autogpt.core.budget.simple import SimpleBudgetManager
from autogpt.core.configuration import Configuration
from autogpt.core.model.language.simple import (
    LanguageModelResponse,
    OpenAILanguageModel,
)
from autogpt.core.planning.simple import ModelPrompt, SimplePlanner
from autogpt.core.plugin.simple import PluginStorageFormat, SimplePluginService
from autogpt.core.workspace import Workspace


class SimpleAgentFactory:
    configuration_defaults = {
        # Which subsystems to use. These must be subclasses of the base classes,
        # but could come from plugins.
        "system": {
            "budget_manager": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.budget.BudgetManager",
            },
            "command_registry": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.command.CommandRegistry",
            },
            "credentials": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.credentials.CredentialsManager",
            },
            "embedding_model": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.model.EmbeddingModel",
            },
            "language_model": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.model.LanguageModel",
            },
            "memory_backend": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.memory.MemoryBackend",
            },
            "planner": {
                "storage_format": PluginStorageFormat.INSTALLED_PACKAGE,
                "storage_route": "autogpt.core.planning.Planner",
            },
        }
    }

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def compile_configuration(
        self, user_configuration: dict
    ) -> Tuple[Configuration, List[str]]:
        """Compile the user's configuration with the defaults."""

        configuration_dict = self.configuration_defaults
        # Copy so we're not mutating the data structure we're looping over.
        system_defaults = self.configuration_defaults["system"].copy()
        # Build up default configuration
        for system_name in system_defaults:
            system_class = self.get_system_class(system_name, user_configuration)
            configuration_dict.update(system_class.configuration_defaults)

        # Apply user overrides
        configuration_dict.update(user_configuration)

        # TODO: Validate the user configuration and compile errors.
        agent_configuration = Configuration(configuration_dict)
        configuration_errors = []
        return agent_configuration, configuration_errors

    def get_system_class(self, system_name: str, configuration: dict | Configuration):
        """Get the system class for the given configuration."""
        if isinstance(configuration, Configuration):
            configuration = configuration.to_dict()

        default_system_location = self.configuration_defaults["system"][system_name]
        system_location = configuration["system"].get(
            system_name, default_system_location
        )
        system_class = SimplePluginService.get_plugin(system_location)

        return system_class

    def get_system_instance(
        self,
        system_name: str,
        configuration: dict | Configuration,
        **kwargs,
    ):
        """Get the system instance for the given configuration."""
        system_class = self.get_system_class(system_name, configuration)
        system_instance = system_class(configuration, **kwargs)
        return system_instance

    def construct_objective_prompt_from_user_input(
        self,
        user_objective: str,
        configuration: Configuration,
    ) -> ModelPrompt:
        agent_planner: SimplePlanner = self.get_system_instance(
            "planner", configuration
        )
        objective_prompt = agent_planner.construct_objective_prompt_from_user_input(
            user_objective,
        )
        return objective_prompt

    async def determine_agent_objective(
        self,
        objective_prompt: ModelPrompt,
        configuration: Configuration,
    ) -> LanguageModelResponse:
        language_model: OpenAILanguageModel = self.get_system_instance(
            "language_model",
            configuration,
            logger=self._logger.getChild("language_model"),
        )
        budget_manager: SimpleBudgetManager = self.get_system_instance(
            "budget_manager",
            configuration,
        )

        model_response = await language_model.determine_agent_objective(
            objective_prompt,
        )
        budget_manager.update_resource_usage_and_cost("openai_budget", model_response)

        return model_response

    def provision_new_agent(self, configuration: Configuration) -> None:
        """Provision a new agent.

        This will create a new workspace and set up all the agent's on-disk resources.

        """
        workspace_class: Type[Workspace] = self.get_system_class(
            "workspace",
            configuration,
        )
        workspace_path = workspace_class.setup_workspace(configuration, self._logger)
        # TODO: Still need to
        #   - Setup the memory backend (create on disk if needed,
        #     otherwise just set configuration)
        #   - Persist all plugins to the workspace directory so they can be loaded
        #     from there on agent startup and avoid any, like version issues and stuff

    def __repr__(self):
        return f"{self.__class__.__name__}()"
