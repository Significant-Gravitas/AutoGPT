from __future__ import annotations

import uuid

from AFAAS.configs.schema import Configurable
from AFAAS.interfaces.adapters.language_model import AbstractLanguageModelProvider

# from AFAAS.interfaces.agent.loop import (  # Import only where it's needed
#     BaseLoop,
#     BaseLoopHook,
# )
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.sdk.logger import AFAASLogger

from .abstract import AbstractAgent

LOG = AFAASLogger(name=__name__)

from AFAAS.core.adapters.openai.chatmodel import AFAASChatOpenAI
from AFAAS.core.workspace.local import AGPTLocalFileWorkspace

# if TYPE_CHECKING:
#     from AFAAS.interfaces.prompts.strategy import (
#         AbstractChatModelResponse,
#         AbstractPromptStrategy,
#     )
#     from AFAAS.interfaces.task.plan import AbstractPlan


class BaseAgent(AbstractAgent, Configurable):

    @property
    def default_llm_provider(self) -> AbstractLanguageModelProvider:
        if self._default_llm_provider is None:
            self._default_llm_provider = AFAASChatOpenAI()
        return self._default_llm_provider

    @property
    def workspace(self) -> AbstractFileWorkspace:
        if self._workspace is None:
            self._workspace = AGPTLocalFileWorkspace(user_id=self.user_id, agent_id=self.agent_id)
        return self._workspace


    class SystemSettings(AbstractAgent.SystemSettings):


        class Config(AbstractAgent.SystemSettings.Config):
            pass



    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    # @classmethod
    # def create_agent(
    #     cls,
    #     agent_settings: BaseAgent.SystemSettings,
    #     memory: AbstractMemory = None,
    #     default_llm_provider: AbstractLanguageModelProvider = None,
    #     workspace: AbstractFileWorkspace = None,
    #     vectorstore: VectorStore = None,  # Optional parameter for custom vectorstore
    #     embedding_model: Embeddings = None,  # Optional parameter for custom embedding model
    # ) -> AbstractAgent:
    #     LOG.info(f"Starting creation of {cls.__name__}")
    #     LOG.trace(f"Debug : Starting creation of  {cls.__module__}.{cls.__name__}")

    #     if not isinstance(agent_settings, cls.SystemSettings):
    #         agent_settings = cls.SystemSettings.parse_obj(agent_settings)

    #     agent = cls.get_instance_from_settings(
    #         agent_settings=agent_settings,
    #         memory = memory ,
    #         default_llm_provider = default_llm_provider,
    #         workspace = workspace,
    #         vectorstore = vectorstore,
    #         embedding_model = embedding_model, 
    #     )

    #     agent_id = agent._create_in_db(agent_settings=agent_settings)
    #     LOG.info(
    #         f"{cls.__name__} id #{agent_id} created in memory. Now, finalizing creation..."
    #     )
    #     # Adding agent_id to the settingsagent_id
    #     agent_settings.agent_id = agent_id

    #     LOG.info(f"Loaded Agent ({agent.__class__.__name__}) with ID {agent_id}")

    #     return agent

    ################################################################################
    ################################ DB INTERACTIONS ################################
    ################################################################################
    def create_agent(
        self
    ) -> str:
        LOG.info(f"Starting creation of {self.__class__.__name__} agent {self.agent_id}")

        agent_table = self.memory.get_table("agents")
        agent_id = agent_table.add(self, id=self.agent_id)
        return agent_id

    # def _create_in_db(
    #     self,
    #     agent_settings: BaseAgent.SystemSettings,
    # ) -> uuid.UUID:
    #     # TODO : Remove the user_id argument

    #     agent_table = self.memory.get_table("agents")
    #     agent_id = agent_table.add(agent_settings, id=agent_settings.agent_id)
    #     return agent_id

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
        page: int = 1,
        page_size: int = 10,
    )  -> list[dict] : #-> list[BaseAgent.SystemSettings]:   
        LOG.trace(f"Entering : {cls.__name__}.list_users_agents_from_memory()")
        from AFAAS.core.db.table.nosql.agent import AgentsTable
        from AFAAS.interfaces.db.db import AbstractMemory
        from AFAAS.interfaces.db.db_table import AbstractTable

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
                        #value=str(cls.__name__),
                        value=str(cls.__module__ + "." + cls.__name__),
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
        from AFAAS.core.db.table.nosql.agent import AgentsTable
        from AFAAS.interfaces.db.db import AbstractMemory

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
