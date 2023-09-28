from __future__ import annotations

import abc
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Dict, Tuple

if TYPE_CHECKING:
    from autogpt.core.agent.base.loop import (  # Import only where it's needed
        BaseLoop,
        BaseLoopHook,
    )

from autogpt.core.agent.base.loop import (
    BaseLoopHook,
)

from autogpt.core.tools.base import ToolsRegistry
from autogpt.core.agent.base.models import (
    BaseAgentConfiguration,
    BaseAgentSettings,
    BaseAgentSystems,
    BaseAgentSystemSettings,
)
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.workspace import Workspace
from autogpt.core.configuration import Configurable


class BaseAgent(abc.ABC):
    CLASS_SYSTEM_SETINGS = BaseAgentSystemSettings
    CLASS_CONFIGURATION = BaseAgentConfiguration
    CLASS_SETTINGS = BaseAgentSettings
    CLASS_SYSTEMS = BaseAgentSystems

    @classmethod
    def get_agent_class(cls) -> Agent:
        return cls

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    @abc.abstractmethod
    def get_agent_from_settings(
        cls,
        agent_settings: BaseAgentSettings,
        logger: logging.Logger,
    ) -> "BaseAgent":
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...

    _loop: BaseLoop = None
    # _loophooks: Dict[str, BaseLoop.LoophooksDict] = {}


class Agent(Configurable, BaseAgent):
    def __init__(
        self,
        settings: BaseAgentSystemSettings,
        logger: logging.Logger,
        memory: Memory,
        workspace: Workspace,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
    ) -> Any:
        self._settings = settings
        self._configuration = settings.configuration
        self._logger = logger
        self._memory = memory
        self._workspace = workspace

        self.user_id = user_id
        self.agent_id = agent_id

        # NOTE : Move to Configurable class ?
        self.agent_class = f"{self.__class__.__name__}"

        # return super().__init__(
        #     self, settings, logger, tool_registry, memory, workspace
        # )

    def add_hook(self, hook: BaseLoopHook, hook_id: uuid.UUID = uuid.uuid4()):
        self._loop._loophooks[hook["name"]][str(hook_id)] = hook

    def remove_hook(self, name: str, hook_id: uuid.UUID) -> bool:
        if name in self._loop._loophooks and hook_id in self._loop._loophooks[name]:
            del self._loop._loophooks[name][hook_id]
            return True
        return False

    async def start(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ) -> None:
        self._logger.info(str(self.__class__) + ".start()")
        return_var = await self._loop.start(
            agent=self,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    async def stop(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ) -> None:
        return_var = await self._loop.stop(
            agent=self,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    async def run(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
        **kwargs,
    ) -> None | dict:
        self._logger.debug(str(self.__class__) + ".run() *kwarg : " + str(kwargs))

        if not self._loop._is_running:
            self._loop._is_running = True
            # Very important, start the loop :-)
            await self.start(
                user_input_handler=user_input_handler,
                user_message_handler=user_message_handler,
            )

            return await self._loop.run(
                agent=self,
                hooks=self._loop._loophooks,
                user_input_handler=user_input_handler,
                user_message_handler=user_message_handler,
                # *kwargs,
            )

        else:
            raise BaseException("Agent Already Running")

    def exit(self, *kwargs) -> None:
        if self._loop._is_running:
            self._loop._is_running = False

    @classmethod
    def get_agent_from_settings(
        cls,
        agent_settings: BaseAgentSettings,
        logger: logging.Logger,
    ) -> Agent:
        # if not isinstance(agent_settings, BaseAgentSettings):
        #     agent_settings: BaseAgentSettings = agent_settings
        if not isinstance(agent_settings, cls.CLASS_SETTINGS):
            agent_settings = cls.CLASS_SETTINGS.parse_obj(agent_settings)
        agent_args = {}

        agent_args["user_id"] = agent_settings.user_id
        agent_args["settings"] = agent_settings.agent
        agent_args["logger"] = logger
        agent_args["workspace"] = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger,
        )

        memory_settings = agent_settings.memory
        agent_args["memory"] = Memory.get_adapter(
            memory_settings=memory_settings, logger=logger
        )

        from importlib import import_module

        module_path, class_name = agent_settings._type_.rsplit(".", 1)
        module = import_module(module_path)
        agent_class = getattr(module, class_name)

        agent_settings, agent_args = agent_class._get_agent_from_settings(
            agent_settings=agent_settings, agent_args=agent_args, logger=logger
        )

        agent = agent_class(**agent_args)

        items = agent_settings.dict().items()
        for key, value in items:
            if key not in agent_class.CLASS_SETTINGS.Config.default_exclude:
                setattr(agent, key, value)

        return agent

    @classmethod
    @abc.abstractmethod
    def _get_agent_from_settings(
        cls, agent_settings: BaseAgentSettings, agent_args: list, logger: logging.Logger
    ) -> Tuple[BaseAgentSettings, list]:
        return agent_settings, agent_args

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
    ) -> BaseAgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing agent system configuration.")
        logger.debug("compile_settings" + str(cls))
        configuration_dict = user_configuration
        configuration_dict["agent"] = cls.build_agent_configuration(
            user_configuration.get("agent", {})
        ).dict()
        # configuration_dict = {
        #     "agent": cls.build_agent_configuration(
        #         user_configuration.get("agent", {})
        #     ).dict(),
        # }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            if system_location is not None and not isinstance(
                system_location, uuid.UUID
            ):
                logger.debug(f"Compiling configuration for system {system_name}")
                system_class = SimplePluginService.get_plugin(system_location)
                system_configuration = user_configuration.get(system_name, {})
                configuration_dict[
                    system_name
                ] = system_class.build_agent_configuration(system_configuration).dict()
            else:
                configuration_dict[system_name] = system_location

        return cls.CLASS_SETTINGS.parse_obj(configuration_dict)

    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    def check_user_context(self, min_len=250, max_len=300):
        pass

    @classmethod
    def create_agent(
        cls,
        agent_settings: BaseAgentSettings,
        logger: logging.Logger,
    ) -> BaseAgent:
        """Create a new agent."""
        logger.info(f"Starting creation of {cls.__name__}")
        logger.debug(f"{cls.__module__}.{cls.__name__}")

        if not isinstance(agent_settings, cls.CLASS_SETTINGS):
            agent_settings = cls.CLASS_SETTINGS.parse_obj(agent_settings)

        agent_id = cls._create_agent_in_memory(
            agent_settings=agent_settings, logger=logger, user_id=agent_settings.user_id
        )

        logger.info(
            f"################################################################################################################################################################################################################################################################################################################################################################"
        )
        logger.info(
            f"{cls.__name__} id #{agent_id} created in memory. Now, finalizing creation..."
        )
        (
            f"##################################################################################################################################################################################################"
        )
        # Adding agent_id to the settings
        agent_settings.agent.agent_id = agent_id
        agent_settings.agent_id = agent_id

        # Processing to custom treatments
        cls._create_agent_custom_treatment(agent_settings=agent_settings, logger=logger)

        logger.debug(agent_settings.__dict__)
        logger.info(
            f"#################################################################################################################################################################################################################################################################################"
        )
        logger.info(f"Loaded Agent ({cls}) with ID {agent_id}")

        agent = cls.get_agent_from_settings(
            agent_settings=agent_settings,
            logger=logger,
        )

        return agent

    @classmethod
    def _create_agent_in_memory(
        cls,
        agent_settings: BaseAgentSettings,
        logger: logging.Logger,
        user_id: uuid.UUID,
    ) -> uuid.UUID:
        agent_settings.agent.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        # TODO : Remove the user_id argument
        # NOTE : Monkey Patching
        BaseAgentSettings.Config.extra = "allow"
        BaseAgentSystemSettings.Config.extra = "allow"
        BaseAgentSystems.Config.extra = "allow"
        BaseAgentConfiguration.Config.extra = "allow"
        agent_settings.agent.configuration.user_id = str(user_id)
        BaseAgentSystems.user_id: uuid.UUID
        agent_settings.user_id = str(user_id)

        from autogpt.core.memory.base import Memory

        memory_settings = agent_settings.memory

        memory = Memory.get_adapter(memory_settings=memory_settings, logger=logger)
        agent_table = memory.get_table("agents")
        agent_id = agent_table.add(agent_settings)
        return agent_id

    def save_agent_in_memory(self) -> uuid.UUID:
        self._logger.debug(self._memory)
        agent_table = self._memory.get_table("agents")
        agent_id = agent_table.update(
            agent_id=self.agent_id, user_id=self.user_id, value=self
        )
        return agent_id

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        agent_settings: BaseAgentSettings,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        # logger.debug("\ncls._get_system_instance : " + str(cls))
        # logger.debug("\n_get_system_instance agent_settings: " + str(agent_settings))
        # logger.debug("\n_get_system_instance system_name: " + str(system_name))
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

    ################################################################################
    ################################ DB INTERACTIONS ################################
    ################################################################################

    @classmethod
    def get_agentsetting_list_from_memory(
        self, user_id: uuid.UUID, logger: logging.Logger
    ) -> list[BaseAgentSettings]:
        from autogpt.core.memory.base import (
            Memory,
            MemoryConfig,
            MemorySettings,
        )
        from autogpt.core.memory.table.base import AgentsTable, BaseTable

        """Warning !!!
        Returns a list of agent settings not a list of agent

        Returns:
            /
        """

        memory_settings = MemorySettings(configuration=MemoryConfig())

        memory = Memory.get_adapter(memory_settings=memory_settings, logger=logger)
        agent_table: AgentsTable = memory.get_table("agents")

        filter = BaseTable.FilterDict(
            {
                "user_id": [
                    BaseTable.FilterItem(
                        value=str(user_id), operator=BaseTable.Operators.EQUAL_TO
                    )
                ],
                "agent_class": [
                    BaseTable.FilterItem(
                        value=str(self.__name__), operator=BaseTable.Operators.EQUAL_TO
                    )
                ],
            }
        )
        agent_list = agent_table.list(filter=filter)
        return agent_list

    @classmethod
    def get_agent_from_memory(
        cls,
        agent_settings: BaseAgentSettings,
        agent_id: uuid.UUID,
        user_id: uuid.UUID,
        logger: logging.Logger,
    ) -> Agent:
        from autogpt.core.memory.base import (
            Memory,
        )
        from autogpt.core.memory.table.base import AgentsTable

        # memory_settings = MemorySettings(configuration=agent_settings.memory)
        memory_settings = agent_settings.memory

        memory = Memory.get_adapter(memory_settings=memory_settings, logger=logger)
        agent_table: AgentsTable = memory.get_table("agents")
        agent = agent_table.get(agent_id=str(agent_id), user_id=str(user_id))

        if not agent:
            return None
        agent = cls.get_agent_from_settings(
            agent_settings=agent_settings,
            logger=logger,
        )
        return agent


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
