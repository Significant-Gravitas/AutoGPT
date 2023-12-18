from __future__ import annotations

import datetime
import importlib
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import yaml
from pydantic import Field, root_validator

from AFAAS.interfaces.agent import BasePromptManager
from AFAAS.interfaces.agent.models import (
    BaseAgentConfiguration, BaseAgentDirectives, BaseAgentSystems)
from AFAAS.interfaces.agent.loop import (  # Import only where it's needed
    BaseLoop, BaseLoopHook)
from AFAAS.configs import (Configurable,
                                                         SystemSettings)
from AFAAS.interfaces.db import AbstractMemory
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.interfaces.adapters.language_model import AbstractLanguageModelProvider

from .abstract import AbstractAgent
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name = __name__)

if TYPE_CHECKING:
    from AFAAS.interfaces.prompts.strategy import (AbstractChatModelResponse, AbstractPromptStrategy)


class BaseAgent(Configurable, AbstractAgent):
    CLASS_CONFIGURATION = BaseAgentConfiguration
    CLASS_SYSTEMS = BaseAgentSystems

    class SystemSettings(AbstractAgent.SystemSettings):
        configuration: BaseAgentConfiguration = BaseAgentConfiguration()

        user_id: str
        agent_id: str = Field(default_factory=lambda: "A" + str(uuid.uuid4()))

        agent_setting_module: Optional[str]
        agent_setting_class: Optional[str]

        #memory: AbstractMemory.SystemSettings = AbstractMemory.SystemSettings()

        class Config(SystemSettings.Config):
            pass

        def dict(self, include_all=False, *args, **kwargs):
            """
            Serialize the object to a dictionary representation.

            Args:
                remove_technical_values (bool, optional): Whether to exclude technical values. Default is True.
                *args: Additional positional arguments to pass to the base class's dict method.
                **kwargs: Additional keyword arguments to pass to the base class's dict method.
                kwargs['exclude'] excludes the fields from the serialization

            Returns:
                dict: A dictionary representation of the object.
            """
            self.prepare_values_before_serialization()  # Call the custom treatment before .dict()
            if not include_all:
                kwargs["exclude"] = self.Config.default_exclude
            # Call the .dict() method with the updated exclude_arg
            return super().dict(*args, **kwargs)

        def json(self, *args, **kwargs):
            """
            Serialize the object to a dictionary representation.

            Args:
                remove_technical_values (bool, optional): Whether to exclude technical values. Default is True.
                *args: Additional positional arguments to pass to the base class's dict method.
                **kwargs: Additional keyword arguments to pass to the base class's dict method.
                kwargs['exclude'] excludes the fields from the serialization

            Returns:
                dict: A dictionary representation of the object.
            """
            LOG.warning(
                "Warning : Recomended use json_api() or json_memory()"
            )
            LOG.warning("BaseAgent.SystemSettings.json()")
            self.prepare_values_before_serialization()  # Call the custom treatment before .json()
            kwargs["exclude"] = self.Config.default_exclude
            return super().json(*args, **kwargs)

        # TODO Implement a BaseSettings class and move it to the BaseSettings ?
        def prepare_values_before_serialization(self):
            self.agent_setting_module = (
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
            self.agent_setting_class = self.__class__.__name__

        # NOTE : To be implemented in the future
        def load_root_values(self, *args, **kwargs):
            pass  # NOTE : Currently not used
            self.agent_name = self.agent.configuration.agent_name
            self.agent_role = self.agent.configuration.agent_role
            self.agent_goals = self.agent.configuration.agent_goals
            self.agent_goal_sentence = self.agent.configuration.agent_goal_sentence

    def __init__(
        self,
        settings: BaseAgent.SystemSettings,
        memory: AbstractMemory,
        workspace: AbstractFileWorkspace,
        prompt_manager: BasePromptManager,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
    ) -> Any:
        self._settings = settings
        self._configuration = settings.configuration
        self._memory = memory
        self._workspace = workspace

        self.user_id = user_id
        self.agent_id = agent_id

        self.settings_agent_class_ = settings.settings_agent_class_
        self.settings_agent_module_ = settings.settings_agent_module_

        self._prompt_manager = prompt_manager
        self._prompt_manager.set_agent(agent=self)

        for key, value in settings.dict():
            if key not in self.SystemSettings.Config.default_exclude:
                if(not hasattr(self, key)):
                    LOG.notice(f"Adding {key} to the agent")
                    setattr(self, key, value)
                else : 
                    LOG.debug(f"{key} set for agent {self.agent_id}")


    def add_hook(self, hook: BaseLoopHook, hook_id: uuid.UUID = uuid.uuid4()):
        """
        Adds a hook to the loop.

        Args:
            hook (BaseLoopHook): The hook to be added.
            hook_id (uuid.UUID, optional): Unique ID for the hook. Defaults to a new UUID.

        Example:
            >>> my_hook = BaseLoopHook(...)
            >>> agent = Agent(...)
            >>> agent.add_hook(my_hook)
        """
        self._loop._loophooks[hook["name"]][str(hook_id)] = hook

    def remove_hook(self, name: str, hook_id: uuid.UUID) -> bool:
        """
        Removes a hook from the loop based on its name and ID.

        Args:
            name (str): Name of the hook.
            hook_id (uuid.UUID): Unique ID of the hook.

        Returns:
            bool: True if removal was successful, otherwise False.

        Example:
            >>> agent = Agent(...)
            >>> removed = agent.remove_hook("my_hook_name", some_uuid)
            >>> print(removed)
            True
        """
        if name in self._loop._loophooks and hook_id in self._loop._loophooks[name]:
            del self._loop._loophooks[name][hook_id]
            return True
        return False

    async def start(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ) -> None:
        LOG.trace(str(self.__class__.__name__) + ".start()")
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

    def exit(self, *kwargs) -> None:
        """
        Exit the agent's loop if it's running.

        Args:
            *kwargs: Additional arguments.

        Example:
            agent = YourClass()
            agent.exit()
        """
        if self._loop._is_running:
            self._loop._is_running = False

    async def run(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
        **kwargs,
    ) -> None | dict:
        """
        Asynchronously run the agent's loop.

        Args:
            user_input_handler (Callable[[str], Awaitable[str]]): Callback for handling user input.
            user_message_handler (Callable[[str], Awaitable[str]]): Callback for handling user messages.
            **kwargs: Additional keyword arguments.

        Returns:
            None | dict: Returns either None or a dictionary based on the loop's run method.

        Raises:
            BaseException: If the agent is already running.

        Example:
            async def input_handler(prompt: str) -> str:
                return input(prompt)

            async def message_handler(message: str) -> str:
                print(message)

            agent = YourClass()
            await agent.run(input_handler, message_handler)
        """
        LOG.trace(
            str(self.__class__.__name__) + ".run() *kwarg : " + str(kwargs)
        )
        self._user_input_handler = user_input_handler
        self._user_message_handler = user_message_handler

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

    @classmethod
    def get_instance_from_settings(
        cls,
        agent_settings: BaseAgent.SystemSettings,
        workspace: AbstractFileWorkspace,
        default_llm_provider: AbstractLanguageModelProvider,
        prompt_manager : BasePromptManager
    ) -> BaseAgent:
        """
        Retrieve an agent instance based on the provided settings and LOG.

        Args:
            agent_settings (BaseAgent.SystemSettings): Configuration settings for the agent.
            logger (logging.Logger): Logger to use for the agent.

        Returns:
            Agent: An agent instance configured according to the provided settings.

        Example:
            logger = logging.getLogger()
            settings = BaseAgent.SystemSettings(user_id="123", ...other_settings...)
            agent = YourClass.get_agent_from_settings(settings)
        """
        if not isinstance(agent_settings, cls.SystemSettings):
            agent_settings = cls.SystemSettings.parse_obj(agent_settings)
            LOG.warning("Warning : agent_settings is not an instance of SystemSettings")

        from importlib import import_module
        module_path, class_name = agent_settings._module_.rsplit(".", 1)
        module = import_module(module_path)
        agent_class: BaseAgent = getattr(module, class_name)

        settings_dict = agent_settings.__dict__
        items = settings_dict.items()

        system_dict: dict[Configurable] = {}
        system_dict["settings"] = agent_settings
        system_dict["user_id"] = agent_settings.user_id
        system_dict["strategies"] = cls.get_strategies()
        system_dict["memory"] = AbstractMemory.get_adapter(
            memory_settings = AbstractMemory.SystemSettings()
        )

        for system, setting in items:
            if system not in ("memory", "tool_registry") and isinstance(
                setting, SystemSettings
            ):
                system_dict[system] = cls._get_system_instance(
                    new_system_name=system,
                    agent_settings=agent_settings,
                    existing_systems=system_dict,
                )

        agent = agent_class(**system_dict 
                            , workspace=workspace, 
                            default_llm_provider=default_llm_provider)



        return agent

    @classmethod
    def _get_system_instance(
        cls,
        new_system_name: str,
        agent_settings: BaseAgent.SystemSettings,
        existing_systems: list,
        *args,
        **kwargs,
    ):
        system_settings: SystemSettings = getattr(agent_settings, new_system_name)
        system_class: Configurable = cls.CLASS_SYSTEMS.load_from_import_path(
            getattr(cls.CLASS_SYSTEMS(), new_system_name)
        )

        if not system_class:
            raise ValueError(
                f"No system class found for {new_system_name} in CLASS_SETTINGS"
            )
        
        if new_system_name == "prompt_manager":
            system_instance = system_class(
                system_settings,
                strategies = existing_systems["strategies"],
                *args,
                agent_systems=existing_systems,
                **kwargs,
            )
            return system_instance
        
        system_instance = system_class(
            system_settings,
            *args,
            agent_systems=existing_systems,
            **kwargs,
        )
        return system_instance

    @classmethod
    def get_strategies(cls) -> list[AbstractPromptStrategy]:

        module = cls.__module__.rsplit('.', 1)[0]
        LOG.trace(f"Entering : {module}.get_strategies()")

        strategies : list[AbstractPromptStrategy] = []

        try:
            # Dynamically import the strategies from the module
            strategies_module = importlib.import_module(f"{module}.strategies")
            # Check if StrategiesSet and get_strategies exist
            if hasattr(strategies_module, 'StrategiesSet') and callable(getattr(strategies_module.StrategiesSet, 'get_strategies', None)):
                strategies = strategies_module.StrategiesSet.get_strategies()
            else:
                LOG.notice(f"{module}.strategies.StrategiesSet or get_strategies method not found")
                raise ImportError("StrategiesSet or get_strategies method not found in the module")

        except ImportError as e:
            LOG.notice(f"Failed to import {module}.strategies: {e}")

        from AFAAS.prompts import load_all_strategies
        strategies += load_all_strategies()
        
        from .strategies.autocorrection import AutoCorrectionStrategy
        from AFAAS.lib.task.rag.afaas_smart_rag import AFAAS_SMART_RAG_Strategy
        common_strategies = [AutoCorrectionStrategy(
                **AutoCorrectionStrategy.default_configuration.dict()
            ),
        AFAAS_SMART_RAG_Strategy(
               **AFAAS_SMART_RAG_Strategy.default_configuration.dict()
        )]

        return  strategies + common_strategies
    

    async def execute_strategy(self, strategy_name: str, **kwargs) -> AbstractChatModelResponse :
        LOG.trace(f"Entering : {self.__class__}.execute_strategy({strategy_name})")
        return await self._prompt_manager._execute_strategy(strategy_name=strategy_name, **kwargs)


    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    @classmethod
    def create_agent(
        cls,
        agent_settings: BaseAgent.SystemSettings,
        workspace: AbstractFileWorkspace,
        default_llm_provider: AbstractLanguageModelProvider,
    ) -> AbstractAgent:
        """
        Create and return a new agent based on the provided settings and LOG.

        Args:
            agent_settings (BaseAgent.SystemSettings): Configuration settings for the agent.
            logger (logging.Logger): Logger to use for the agent.

        Returns:
            BaseAgent: An agent instance configured according to the provided settings.

        Example:
            logger = logging.getLogger()
            settings = BaseAgent.SystemSettings(user_id="123", ...other_settings...)
            agent = YourClass.create_agent(settings)
        """
        LOG.info(f"Starting creation of {cls.__name__}")
        LOG.trace(f"Debug : Starting creation of  {cls.__module__}.{cls.__name__}")

        if not isinstance(agent_settings, cls.SystemSettings):
            agent_settings = cls.SystemSettings.parse_obj(agent_settings)

        agent_id = cls._create_in_db(agent_settings=agent_settings)
        LOG.info(
            f"{cls.__name__} id #{agent_id} created in memory. Now, finalizing creation..."
        )
        # Adding agent_id to the settingsagent_id
        agent_settings.agent_id = agent_id

        # Processing to custom treatments
        cls._create_agent_custom_treatment(agent_settings=agent_settings)

        LOG.info(f"Loaded Agent ({cls.__name__}) with ID {agent_id}")

        agent = cls.get_instance_from_settings(
            agent_settings=agent_settings,
            workspace=workspace,
            default_llm_provider=default_llm_provider,
        )

        return agent

    @classmethod
    @abstractmethod
    def _create_agent_custom_treatment(
        cls, agent_settings: BaseAgent.SystemSettings
    ) -> None:
        pass

    ################################################################################
    ################################ DB INTERACTIONS ################################
    ################################################################################

    @classmethod
    def _create_in_db(
        cls,
        agent_settings: BaseAgent.SystemSettings,
    ) -> uuid.UUID:
        # TODO : Remove the user_id argument
        # NOTE : Monkey Patching
        # BaseAgent.SystemSettings.Config.extra = "allow"
        # # BaseAgentSystems.Config.extra = "allow"
        # BaseAgentConfiguration.Config.extra = "allow"

        from AFAAS.interfaces.db import AbstractMemory

        memory_settings = agent_settings.memory

        memory = AbstractMemory.get_adapter(
            memory_settings=memory_settings
        )
        agent_table = memory.get_table("agents")
        agent_id = agent_table.add(agent_settings, id=agent_settings.agent_id)
        return agent_id

    def save_agent_in_memory(self) -> uuid.UUID:
        LOG.trace(self._memory)
        agent_table = self._memory.get_table("agents")
        agent_id = agent_table.update(
            agent_id=self.agent_id, user_id=self.user_id, value=self
        )
        return agent_id

    @classmethod
    def list_users_agents_from_memory(
        cls,
        user_id: uuid.UUID,
        #workspace: AbstractFileWorkspace,
        page: int = 1,
        page_size: int = 10,
    )  -> list[dict] : #-> list[BaseAgent.SystemSettings]:   
        """
        Fetch a list of agent settings from memory based on the user ID.

        Args:
            user_id (uuid.UUID): The unique identifier for the user.
            logger (logging.Logger): Logger to use.

        Returns:
            list[BaseAgent.SystemSettings]: List of agent settings from memory.

        Example:
            logger = logging.getLogger()
            user_id = uuid.uuid4()
            agent_settings_list = YourClass.get_agentsetting_list_from_memory(user_id)
            print(agent_settings_list)
        """
        LOG.trace(f"Entering : {cls.__name__}.list_users_agents_from_memory()")
        from AFAAS.interfaces.db import AbstractMemory
        from AFAAS.interfaces.db_table import AbstractTable
        from AFAAS.core.db.table import AgentsTable

        memory_settings = AbstractMemory.SystemSettings()

        memory = AbstractMemory.get_adapter(
            memory_settings=memory_settings
        )
        agent_table: AgentsTable = memory.get_table("agents")

        filter = AbstractTable.FilterDict(
            {
                "user_id": [
                    AbstractTable.FilterItem(
                        value=str(user_id), operator=AbstractTable.Operators.EQUAL_TO
                    )
                ],
                AbstractAgent.SystemSettings.Config.AGENT_CLASS_FIELD_NAME: [
                    AbstractTable.FilterItem(
                        value=str(cls.__name__),
                        operator=AbstractTable.Operators.EQUAL_TO,
                    )
                ],
            }
        )

        agent_list: list[dict] = agent_table.list(filter=filter)
        return agent_list
        agent_settings_list: list[cls.SystemSettings] = []

        # TODO : Move to Table
        for agent in agent_list:
            agent["workspace"] = workspace
            agent_settings_list.append(cls.SystemSettings(**agent))

        return agent_settings_list

    @classmethod
    def get_agent_from_memory(
        cls,
        agent_settings: BaseAgent.SystemSettings,
        agent_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> BaseAgent:
        from AFAAS.interfaces.db import AbstractMemory
        from AFAAS.core.db.table import AgentsTable

        # memory_settings = Memory.SystemSettings(configuration=agent_settings.memory)
        memory_settings = agent_settings.memory

        memory = AbstractMemory.get_adapter(
            memory_settings=memory_settings
        )
        agent_table: AgentsTable = memory.get_table("agents")
        agent_dict_from_db = agent_table.get(
            agent_id=str(agent_id), user_id=str(user_id)
        )

        if not agent_dict_from_db:
            return None

        agent = cls.get_instance_from_settings(
            agent_settings=agent_settings.copy(update=agent_dict_from_db),
        )
        return agent

    @classmethod
    @abstractmethod
    def load_prompt_settings(
        cls, erase=False, file_path: str = ""
    ) -> BaseAgentDirectives:
        # Get the directory containing the current class file
        base_agent_dir = os.path.dirname(__file__)
        # Construct the path to the YAML file based on __file__
        current_settings_path = os.path.join(base_agent_dir, "prompt_settings.yaml")

        settings = {}
        # Load settings from the current directory (based on __file__)
        if os.path.exists(current_settings_path):
            with open(current_settings_path, "r") as file:
                settings = yaml.load(file, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError(f"Can't locate file {current_settings_path}")

        agent_directives = BaseAgentDirectives(
            constraints=settings.get("constraints", []),
            resources=settings.get("resources", []),
            best_practices=settings.get("best_practices", []),
        )

        # Load settings from the specified directory (based on 'file')
        if file_path:
            specified_settings_path = os.path.join(
                os.path.dirname(file_path), "prompt_settings.yaml"
            )

            if os.path.exists(specified_settings_path):
                with open(specified_settings_path, "r") as file_path:
                    specified_settings = yaml.safe_load(file_path)
                    for key, items in specified_settings.items():
                        if key not in agent_directives.keys():
                            agent_directives[key]: list[str] = items
                        else:
                            # If the item already exists, update it with specified_settings
                            if erase:
                                agent_directives[key] = items
                            else:
                                agent_directives[key] += items

        return agent_directives


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty dictionaries at the leaves.

    Args:
        d (dict): The dictionary to prune.

    Returns:
        dict: The pruned dictionary.

    Example:
        input_dict = {"a": {}, "b": {"c": {}, "d": "value"}}
        pruned_dict = _prune_empty_dicts(input_dict)
        print(pruned_dict)  # Expected: {"b": {"d": "value"}}
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
