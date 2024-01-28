from __future__ import annotations

import datetime
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, Optional

from pydantic import Field

from AFAAS.configs.schema import Configurable, SystemSettings
from AFAAS.interfaces.adapters.language_model import AbstractLanguageModelProvider
from AFAAS.interfaces.agent.assistants.prompt_manager import BasePromptManager
from AFAAS.interfaces.agent.loop import BaseLoop  # Import only where it's needed
from AFAAS.interfaces.db.db import AbstractMemory
from AFAAS.interfaces.workflow import WorkflowRegistry
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.message_common import AFAASMessageStack
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain.vectorstores import VectorStore

if TYPE_CHECKING:
    from AFAAS.interfaces.prompts.strategy import AbstractChatModelResponse
    from AFAAS.interfaces.task.plan import AbstractPlan


class AbstractAgent(ABC):

    _agent_type_: ClassVar[str] = __name__
    _agent_module_: ClassVar[str] = __module__ + "." + __name__
    plan : Optional[AbstractPlan] = None

    @property
    def vectorstores(self) -> dict[str , VectorStore]:
        # Ensure 'tasks' and 'documents' VectorStores are initialized
        self._ensure_vectorstore_initialized("tasks")
        self._ensure_vectorstore_initialized("documents")
        return self._vectorstores

    def _ensure_vectorstore_initialized(self, key: str):
        if key not in self._vectorstores or self._vectorstores[key] is None:
            self._vectorstores[key] = Chroma(
                persist_directory=f'data/chroma/{key}',
                embedding_function=self.embedding_model
            )

    @vectorstores.setter
    def vectorstores(self, value: dict[str , VectorStore]):
        for key, vectorstore in value.items():
            self._vectorstores[key] = vectorstore

    @property
    def embedding_model(self) -> Embeddings:
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings()
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, value : Embeddings):
        self._embedding_model = value

    @property
    def db(self) -> AbstractMemory:
        if self._db is None:
            self._db = AbstractMemory.get_adapter()
        return self._db

    @db.setter
    def db(self, value: AbstractMemory):
        self._db = value

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


    @property
    def workflow_registry(self) -> WorkflowRegistry:
        if self._workflow_registry is None:
            self._workflow_registry = WorkflowRegistry()
        return self._workflow_registry

    class SystemSettings(SystemSettings):


        class Config(SystemSettings.Config):
            AGENT_CLASS_FIELD_NAME : str = "_type_"
            AGENT_CLASS_MODULE_NAME : str = "_module_"

        modified_at: datetime.datetime = datetime.datetime.now()
        created_at: datetime.datetime = datetime.datetime.now()
        agent_name: str = Field(default="New Agent")
        agent_goal_sentence: Optional[str]
        agent_goals: Optional[list]
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
                #FIXME:0.0.2 Implement same behaviour for TaskStack in AbstractBaseTask
                if isinstance(field_type, type) and field_name in data and issubclass(field_type, AFAASMessageStack):
                    setattr(self, field_name, AFAASMessageStack(_messages=data[field_name]))

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
                "Warning : Recomended use json_api() or json_db()"
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
        db: AbstractMemory,
        workspace: AbstractFileWorkspace,
        prompt_manager: BasePromptManager,
        default_llm_provider: AbstractLanguageModelProvider,
        vectorstores: dict[str , VectorStore],
        embedding_model : Embeddings,
        workflow_registry: WorkflowRegistry,
        user_id: str,
        agent_id: str,
        **kwargs,
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

        self._db : AbstractMemory = db

        self._workspace : AbstractFileWorkspace = workspace
        self.workspace.initialize()

        self._default_llm_provider : AbstractLanguageModelProvider = default_llm_provider
        self._embedding_model : Embeddings = embedding_model
        self._vectorstores : dict[VectorStore] = {}
        for key, vectorstore in vectorstores.items():
            self._vectorstores[key] : VectorStore = vectorstore

        self._workflow_registry : WorkflowRegistry = workflow_registry

        self._loop : BaseLoop = None

        for key, value in settings.dict().items():
            if key not in self.SystemSettings.Config.default_exclude:
                if(not hasattr(self, key)):
                    LOG.notice(f"Adding {key} to the agent")
                    setattr(self, key, value)
                else : 
                    LOG.debug(f"{key} set for agent {self.agent_id}")

        LOG.trace(f"{self.__class__.__name__}.__init__() : Leaving")

    # @classmethod
    # def get_instance_from_settings(
    #     cls,
    #     agent_settings: AbstractAgent.SystemSettings,
    #     db: AbstractMemory = None,
    #     default_llm_provider: AbstractLanguageModelProvider = None,
    #     workspace: AbstractFileWorkspace = None,
    #     vectorstores: dict[VectorStore] = None,  # Optional parameter for custom vectorstore
    #     embedding_model: Embeddings = None,  
    # ) -> AbstractAgent:
    #     if not isinstance(agent_settings, cls.SystemSettings):
    #         agent_settings = cls.SystemSettings.parse_obj(agent_settings)
    #         LOG.warning("Warning : agent_settings is not an instance of SystemSettings")

    #     # TODO: Just pass **agent_settings.dict() to the constructor
    #     system_dict: dict[Configurable] = {}
    #     system_dict["settings"] = agent_settings
    #     system_dict["user_id"] = agent_settings.user_id
    #     system_dict["agent_id"] = agent_settings.agent_id

    #     agent = cls(    **system_dict , 
    #                     workspace=workspace, 
    #                     default_llm_provider=default_llm_provider,
    #                     vectorstores=vectorstores,
    #                     embedding_model=embedding_model,
    #                     db=db,
    #                     )

    #     return agent


    # @classmethod
    # async def db_get(
    #     cls,
    #     agent_settings: AbstractAgent.SystemSettings,
    #     agent_id: str,
    #     user_id: str,
    # ) -> AbstractAgent:
    #     from AFAAS.core.db.table.nosql.agent import AgentsTable
    #     from AFAAS.interfaces.db.db import AbstractMemory

    #     db_settings = agent_settings.db

    #     db = AbstractMemory.get_adapter(
    #         db_settings=db_settings
    #     )
    #     agent_table: AgentsTable = await db.get_table("agents")
    #     agent_dict_from_db = await agent_table.get(
    #         agent_id=str(agent_id), user_id=str(user_id)
    #     )

    #     if not agent_dict_from_db:
    #         return None

    #     agent = cls.get_instance_from_settings(
    #         agent_settings=agent_settings.copy(update=agent_dict_from_db),
    #     )
    #     return agent

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


    async def execute_strategy(self, strategy_name: str, **kwargs) -> AbstractChatModelResponse :
        LOG.trace(f"Entering : {self.__class__}.execute_strategy({strategy_name})")
        return await self._prompt_manager._execute_strategy(strategy_name=strategy_name, **kwargs)


AbstractAgent.SystemSettings.update_forward_refs()
