from __future__ import annotations

import datetime
import importlib
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import yaml
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import Field, root_validator

from AFAAS.configs import Configurable, SystemSettings
from AFAAS.interfaces.adapters.language_model import AbstractLanguageModelProvider
from AFAAS.interfaces.agent import BasePromptManager
from AFAAS.interfaces.agent.loop import (  # Import only where it's needed
    BaseLoop,
    BaseLoopHook,
)
from AFAAS.interfaces.db import AbstractMemory

from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.sdk.logger import AFAASLogger

from .abstract import AbstractAgent

LOG = AFAASLogger(name = __name__)
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from AFAAS.core.adapters.openai import AFAASChatOpenAI
from AFAAS.core.workspace.local import AGPTLocalFileWorkspace

if TYPE_CHECKING:
    from AFAAS.interfaces.prompts.strategy import (
        AbstractChatModelResponse,
        AbstractPromptStrategy,
    )
    from AFAAS.interfaces.task import AbstractPlan


class BaseAgent(Configurable, AbstractAgent):

    plan : Optional[AbstractPlan] = None

    @property
    def vectorstore(self) -> VectorStore:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory='data/chroma',
                embedding_function=self.embedding_model
            )
        return self._vectorstore

    @vectorstore.setter
    def vectorstore(self, value : VectorStore):
        self._vectorstore = value

    @property
    def embedding_model(self) -> Embeddings:
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings()
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value : Embeddings):
        self._embedding_model = value

    @property
    def memory(self) -> AbstractMemory:
        if self._memory is None:
            self._memory = AbstractMemory.get_adapter()
        return self._memory

    @memory.setter
    def memory(self, value: AbstractMemory):
        self._memory = value

    @property
    def default_llm_provider(self) -> AbstractLanguageModelProvider:
        if self._default_llm_provider is None:
            self._default_llm_provider = AFAASChatOpenAI()
        return self._default_llm_provider

    @default_llm_provider.setter
    def default_llm_provider(self, value: AbstractLanguageModelProvider):
        self._default_llm_provider = value

    @property
    def workspace(self) -> AbstractFileWorkspace:
        if self._workspace is None:
            self._workspace = AGPTLocalFileWorkspace(user_id=self.user_id, agent_id=self.agent_id)
        return self._workspace

    @workspace.setter
    def workspace(self, value: AbstractFileWorkspace):
        self._workspace = value

    class SystemSettings(AbstractAgent.SystemSettings):

        user_id: str
        agent_id: str = Field(default_factory=lambda: BaseAgent.SystemSettings.generate_uuid())

        @staticmethod
        def generate_uuid():
            return "A" + str(uuid.uuid4())

        agent_setting_module: Optional[str]
        agent_setting_class: Optional[str]

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

    def __init__(
        self,
        settings: BaseAgent.SystemSettings,
        memory: AbstractMemory,
        workspace: AbstractFileWorkspace,
        prompt_manager: BasePromptManager,
        default_llm_provider: AbstractLanguageModelProvider,
        vectorstore: VectorStore,
        embedding_model : Embeddings,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
    ) -> Any:
        LOG.trace(f"{self.__class__.__name__}.__init__() : Entering")
        self._settings = settings

        self.agent_id = agent_id
        self.user_id = user_id
        self.agent_name = settings.agent_name

        #
        # Step 1 : Set the chat model provider
        #
        self.settings_agent_class_ = settings.settings_agent_class_
        self.settings_agent_module_ = settings.settings_agent_module_

        self._prompt_manager : BasePromptManager = prompt_manager
        self._prompt_manager.set_agent(agent=self)

        self._memory : AbstractMemory = memory
        self._workspace : AbstractFileWorkspace = workspace
        self.workspace.initialize()
        self._default_llm_provider : AbstractLanguageModelProvider = default_llm_provider
        self._vectorstore : VectorStore = vectorstore
        self._embedding_model : Embeddings = embedding_model

        for key, value in settings.dict().items():
            if key not in self.SystemSettings.Config.default_exclude:
                if(not hasattr(self, key)):
                    LOG.notice(f"Adding {key} to the agent")
                    setattr(self, key, value)
                else : 
                    LOG.debug(f"{key} set for agent {self.agent_id}")

        LOG.trace(f"{self.__class__.__name__}.__init__() : Leaving")


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
        memory: AbstractMemory = None,
        default_llm_provider: AbstractLanguageModelProvider = None,
        workspace: AbstractFileWorkspace = None,
        vectorstore: VectorStore = None,  # Optional parameter for custom vectorstore
        embedding_model: Embeddings = None,  
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

        settings_dict = agent_settings.__dict__
        items = settings_dict.items()

        system_dict: dict[Configurable] = {}
        system_dict["settings"] = agent_settings
        system_dict["user_id"] = agent_settings.user_id
        system_dict["agent_id"] = agent_settings.agent_id

        agent = cls(**system_dict , 
                            workspace=workspace, 
                            default_llm_provider=default_llm_provider,
                            vectorstore=vectorstore,
                            embedding_model=embedding_model,
                            memory=memory,
                            )

        return agent


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
        memory: AbstractMemory = None,
        default_llm_provider: AbstractLanguageModelProvider = None,
        workspace: AbstractFileWorkspace = None,
        vectorstore: VectorStore = None,  # Optional parameter for custom vectorstore
        embedding_model: Embeddings = None,  # Optional parameter for custom embedding model
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

        agent = cls.get_instance_from_settings(
            agent_settings=agent_settings,
            memory = memory ,
            default_llm_provider = default_llm_provider,
            workspace = workspace,
            vectorstore = vectorstore,
            embedding_model = embedding_model, 
        )

        agent_id = agent._create_in_db(agent_settings=agent_settings)
        LOG.info(
            f"{cls.__name__} id #{agent_id} created in memory. Now, finalizing creation..."
        )
        # Adding agent_id to the settingsagent_id
        agent_settings.agent_id = agent_id

        LOG.info(f"Loaded Agent ({agent.__class__.__name__}) with ID {agent_id}")

        return agent

    ################################################################################
    ################################ DB INTERACTIONS ################################
    ################################################################################

    def _create_in_db(
        self,
        agent_settings: BaseAgent.SystemSettings,
    ) -> uuid.UUID:
        # TODO : Remove the user_id argument

        agent_table = self.memory.get_table("agents")
        agent_id = agent_table.add(agent_settings, id=agent_settings.agent_id)
        return agent_id

    def save_agent_in_memory(self) -> str:
        LOG.trace(self.memory)
        agent_table = self.memory.get_table("agents")
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
        from AFAAS.core.db.table import AgentsTable
        from AFAAS.interfaces.db import AbstractMemory
        from AFAAS.interfaces.db_table import AbstractTable

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


    @classmethod
    def get_agent_from_memory(
        cls,
        agent_settings: BaseAgent.SystemSettings,
        agent_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> BaseAgent:
        from AFAAS.core.db.table import AgentsTable
        from AFAAS.interfaces.db import AbstractMemory

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
