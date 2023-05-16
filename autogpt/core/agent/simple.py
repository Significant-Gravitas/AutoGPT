import logging
from pathlib import Path

from pydantic import BaseModel

from autogpt.core.agent.base import Agent
from autogpt.core.command import CommandRegistrySettings, SimpleCommandRegistry
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.model import (
    EmbeddingModelSettings,
    LanguageModelResponse,
    LanguageModelSettings,
    SimpleEmbeddingModel,
    SimpleLanguageModel,
)
from autogpt.core.planning.simple import ModelPrompt, PlannerSettings, SimplePlanner
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource.model_providers import OpenAIProvider, OpenAISettings
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings


class AgentSystems(SystemConfiguration):
    command_registry: PluginLocation
    memory: PluginLocation
    embedding_model: PluginLocation
    language_model: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    command_registry: CommandRegistrySettings
    memory: MemorySettings
    embedding_model: EmbeddingModelSettings
    language_model: LanguageModelSettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings


class SimpleAgent(Agent, Configurable):
    defaults = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            systems=AgentSystems(
                command_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.command.SimpleCommandRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                embedding_model=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.model.SimpleEmbeddingModel",
                ),
                language_model=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.model.SimpleLanguageModel",
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
        configuration: AgentConfiguration,
        logger: logging.Logger,
        command_registry: SimpleCommandRegistry,
        memory: SimpleMemory,
        embedding_model: SimpleEmbeddingModel,
        language_model: SimpleLanguageModel,
        openai_provider: OpenAIProvider,
        planning: SimplePlanner,
        workspace: SimpleWorkspace,
    ):
        self._configuration = configuration
        self._logger = logger
        self._command_registry = command_registry
        self._memory = memory
        self._embedding_model = embedding_model
        self._language_model = language_model
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._planning = planning
        self._workspace = workspace

    def step(self, *args, **kwargs):
        pass

    def run(self):
        pass

    @staticmethod
    def load_system(
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        **kwargs,
    ):
        logger.debug(f"Loading system {system_name}")
        system_load_config = agent_settings.system[system_name]
        system_class = SimplePluginService.get_plugin(system_load_config)

        system_logger = logger.getChild(system_name)
        system_config = getattr(agent_settings, system_name)
        return system_class(system_config, system_logger, **kwargs)

    def __repr__(self):
        return "SimpleAgent()"

    ###############################################################
    # Factory interface for agent boostrapping and initialization #
    ###############################################################

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Compiling agent settings")
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.process_user_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.process_user_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    def construct_objective_prompt_from_user_objective(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> ModelPrompt:
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
        )
        objective_prompt = agent_planner.construct_objective_prompt_from_user_input(
            user_objective,
        )
        return objective_prompt

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        objective_prompt: ModelPrompt,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> LanguageModelResponse:
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger=logger,
        )

        language_model: SimpleLanguageModel = cls._get_system_instance(
            "language_model",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )

        model_response = await language_model.determine_agent_objective(
            objective_prompt,
        )

        return model_response

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
