from __future__ import annotations

import datetime
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, Optional

import yaml
from pydantic import Field, root_validator

from AFAAS.configs import SystemSettings, Configurable
from AFAAS.interfaces.adapters.language_model import AbstractLanguageModelProvider
from AFAAS.interfaces.agent import BasePromptManager
from AFAAS.interfaces.agent.loop import BaseLoop  # Import only where it's needed
from AFAAS.interfaces.db import AbstractMemory
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.message_agent_agent import MessageAgentAgent
from AFAAS.lib.message_agent_llm import MessageAgentLLM
from AFAAS.lib.message_agent_user import MessageAgentUser
from AFAAS.lib.message_common import AFAASMessage, AFAASMessageStack
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from AFAAS.interfaces.task import AbstractPlan
    from AFAAS.interfaces.prompts.strategy import (
        AbstractChatModelResponse,
        AbstractPromptStrategy,
    )


class AbstractAgent(ABC):

    _agent_type_: ClassVar[str] = __name__
    _agent_module_: ClassVar[str] = __module__ + "." + __name__
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
    @abstractmethod
    def default_llm_provider(self) -> AbstractLanguageModelProvider:
        ...

    @default_llm_provider.setter
    def default_llm_provider(self, value: AbstractLanguageModelProvider):
        self._default_llm_provider = value

    @property
    @abstractmethod
    def workspace(self) -> AbstractFileWorkspace:
        ...

    @workspace.setter
    def workspace(self, value: AbstractFileWorkspace):
        self._workspace = value


    class SystemSettings(SystemSettings):


        class Config(SystemSettings.Config):
            AGENT_CLASS_FIELD_NAME : str = "_type_"
            AGENT_CLASS_MODULE_NAME : str = "_module_"

        modified_at: datetime.datetime = datetime.datetime.now()
        created_at: datetime.datetime = datetime.datetime.now()
        user_id: str

        agent_id: str = Field(default_factory=lambda: AbstractAgent.SystemSettings.generate_uuid())
        @staticmethod
        def generate_uuid():
            return "A" + str(uuid.uuid4())

        _message_agent_user: Optional[AFAASMessageStack] = Field(default=[])
        @property
        def message_agent_user(self) -> AFAASMessageStack:
            if self._message_agent_user is None:
                self._message_agent_user = AFAASMessageStack(
                    parent_task=self, description="message_agent_user"
                )
            return self._message_agent_user

        def __init__(self, **data):
            super().__init__(**data)
            for field_name, field_type in self.__annotations__.items():
                # Check if field_type is a class before calling issubclass
                if isinstance(field_type, type) and field_name in data and issubclass(field_type, AFAASMessageStack):
                    setattr(self, field_name, AFAASMessageStack(_stack=data[field_name]))

        @property
        def _type_(self):
            # == "".join(self.__class__.__qualname__.split(".")[:-1])  
            return self.__class__.__qualname__.split(".")[0]    

        @property
        def _module_(self):
            # Nested Class
            return self.__module__ + "." + self._type_

        # agent_setting_module: Optional[str]
        # agent_setting_class: Optional[str]

        @classmethod
        @property
        def settings_agent_class_(cls):
            return cls.__qualname__.partition(".")[0]

        @classmethod
        @property
        def settings_agent_module_(cls):
            return cls.__module__ + "." + ".".join(cls.__qualname__.split(".")[:-1])

        def dict(self, include_all=False, *args, **kwargs):
            self.prepare_values_before_serialization()  # Call the custom treatment before .dict()
            if not include_all:
                kwargs["exclude"] = self.Config.default_exclude
            # Call the .dict() method with the updated exclude_arg
            return super().dict(*args, **kwargs)

        def json(self, *args, **kwargs):
            LOG.warning(
                "Warning : Recomended use json_api() or json_memory()"
            )
            LOG.warning("AbstractAgent.SystemSettings.json()")
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
        settings: AbstractAgent.SystemSettings,
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

        self._loop : BaseLoop = None

        for key, value in settings.dict().items():
            if key not in self.SystemSettings.Config.default_exclude:
                if(not hasattr(self, key)):
                    LOG.notice(f"Adding {key} to the agent")
                    setattr(self, key, value)
                else : 
                    LOG.debug(f"{key} set for agent {self.agent_id}")

        LOG.trace(f"{self.__class__.__name__}.__init__() : Leaving")


    # def add_hook(self, hook: BaseLoopHook, hook_id: uuid.UUID = uuid.uuid4()):
    #     self._loop._loophooks[hook["name"]][str(hook_id)] = hook

    # def remove_hook(self, name: str, hook_id: uuid.UUID) -> bool:
    #     if name in self._loop._loophooks and hook_id in self._loop._loophooks[name]:
    #         del self._loop._loophooks[name][hook_id]
    #         return True
    #     return False

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
        if self._loop._is_running:
            self._loop._is_running = False

    async def run(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
        **kwargs,
    ) -> None | dict:
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
        agent_settings: AbstractAgent.SystemSettings,
        memory: AbstractMemory = None,
        default_llm_provider: AbstractLanguageModelProvider = None,
        workspace: AbstractFileWorkspace = None,
        vectorstore: VectorStore = None,  # Optional parameter for custom vectorstore
        embedding_model: Embeddings = None,  
    ) -> AbstractAgent:
        """
        Retrieve an agent instance based on the provided settings and LOG.

        Args:
            agent_settings (AbstractAgent.SystemSettings): Configuration settings for the agent.
            logger (logging.Logger): Logger to use for the agent.

        Returns:
            Agent: An agent instance configured according to the provided settings.

        Example:
            logger = logging.getLogger()
            settings = AbstractAgent.SystemSettings(user_id="123", ...other_settings...)
            agent = YourClass.get_agent_from_settings(settings)
        """
        if not isinstance(agent_settings, cls.SystemSettings):
            agent_settings = cls.SystemSettings.parse_obj(agent_settings)
            LOG.warning("Warning : agent_settings is not an instance of SystemSettings")

        # TODO: Just pass **agent_settings.dict() to the constructor
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


AbstractAgent.SystemSettings.update_forward_refs()
