from __future__ import annotations

import logging
from typing import List, Tuple, Type

from autogpt.core.model import LanguageModelResponse, SimpleLanguageModel
from autogpt.core.planning.simple import ModelPrompt, SimplePlanner
from autogpt.core.plugin.simple import PluginStorageFormat, SimplePluginService
from autogpt.core.workspace import Workspace


class SimpleAgentFactory:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

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
        language_model: SimpleLanguageModel = self.get_system_instance(
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
