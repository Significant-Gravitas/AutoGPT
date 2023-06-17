import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autogpt.core.agent.base import Agent
from autogpt.core.ability import AbilityRegistrySettings, SimpleAbilityRegistry
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.embedding import EmbeddingModelSettings, SimpleEmbeddingModel
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.planning import LanguageModelResponse, PlannerSettings, SimplePlanner
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource.model_providers import OpenAIProvider, OpenAISettings
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    embedding_model: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    creation_time: str
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    embedding_model: EmbeddingModelSettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        self.planning.configuration.agent_name = agent_goals["agent_name"]
        self.planning.configuration.agent_role = agent_goals["agent_role"]
        self.planning.configuration.agent_goals = agent_goals["agent_goals"]


class SimpleAgent(Agent, Configurable):
    defaults = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            cycle_count=0,
            creation_time="",
            systems=AgentSystems(
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.ability.SimpleAbilityRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                embedding_model=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.embedding.SimpleEmbeddingModel",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.resource.model_providers.OpenAIProvider",
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
        ),
    )

    def __init__(
        self,
        settings: AgentSystemSettings,
        logger: logging.Logger,
        ability_registry: SimpleAbilityRegistry,
        memory: SimpleMemory,
        embedding_model: SimpleEmbeddingModel,
        openai_provider: OpenAIProvider,
        planning: SimplePlanner,
        workspace: SimpleWorkspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._ability_registry = ability_registry
        self._memory = memory
        self._embedding_model = embedding_model
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._planning = planning
        self._workspace = workspace

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "Agent":
        agent_settings = SimpleWorkspace.load_agent_settings(workspace_path)
        agent_args = {}

        agent_args["settings"] = agent_settings.agent
        agent_args["logger"] = logger
        agent_args["workspace"] = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger,
        )
        agent_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger,
        )
        agent_args["embedding_model"] = cls._get_system_instance(
            "embedding_model",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
        )
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
        )

        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
        )
        agent_args["memory"] = cls._get_system_instance(
            "memory",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
        )

        return cls(**agent_args)

    async def step(self, user_feedback: str, *args, **kwargs):
        self._configuration.cycle_count += 1
        return await self._planning.make_initial_plan()


    def __repr__(self):
        return "SimpleAgent()"

    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        configuration_dict = {
            "agent": cls.get_user_config(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.build_agent_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.build_agent_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> LanguageModelResponse:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger=logger,
        )
        logger.debug("Loading agent planner.")
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        logger.debug("determining agent name and goals.")
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response

    @classmethod
    def provision_agent(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ):
        agent_settings.agent.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: SimpleWorkspace = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger=logger,
        )
        return workspace.setup_workspace(agent_settings, logger)

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        system_locations = agent_settings.agent.configuration.systems.dict()

        system_settings = getattr(agent_settings, system_name)
        system_class = SimplePluginService.get_plugin(system_locations[system_name])
        system_instance = system_class(
            system_settings,
            *args,
            logger=logger.getChild(system_name),
            **kwargs,
        )
        return system_instance


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty dictionaries at the leaves.

    Args:
        d: The dictionary to prune.

    Returns:
        The pruned dictionary.
    """
    pruned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            pruned_value = _prune_empty_dicts(value)
            if (
                pruned_value
            ):  # if the pruned dictionary is not empty, add it to the result
                pruned[key] = pruned_value
        else:
            pruned[key] = value
    return pruned
