from __future__ import annotations

from AFAAS.configs.schema import AFAASModel, Configurable
from AFAAS.interfaces.adapters.language_model import AbstractLanguageModelProvider
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.sdk.logger import AFAASLogger

from .abstract import AbstractAgent

LOG = AFAASLogger(name=__name__)

from AFAAS.core.adapters.openai.chatmodel import AFAASChatOpenAI
from AFAAS.core.workspace.local import AGPTLocalFileWorkspace


class BaseAgent(AFAASModel , AbstractAgent, Configurable):

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

        # TODO[pydantic]: The `Config` class inherits from another class, please create the `model_config` manually.
        # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
        class Config(AbstractAgent.SystemSettings.Config):
            pass
    class Config(AFAASModel.Config):
        default_exclude = set(AbstractAgent.SystemSettings.Config.default_exclude ) | {
            'embedding_model', 
            'settings',
            'vectorstores', 
            'workflow_registry', 
            'db', 
            'default_llm_provider', 
            'prompt_manager', 
            'workspace', 
            '_settings', 
            '_prompt_manager', 
            '_db', 
            '_workspace', 
            '_default_llm_provider', 
            '_embedding_model', 
            '_vectorstores', 
            '_workflow_registry', 
            '_loop', 
            '_tool_registry', 
            '_tool_executor', 
        }

    def __init__(self, **kwargs):
        #super().__init__(**kwargs)
        AFAASModel.__init__(self, **kwargs)
        Configurable.__init__(self, **kwargs)
        AbstractAgent.__init__(self, **kwargs)

    ################################################################################
    ################################ DB INTERACTIONS ################################
    ################################################################################
    async def db_create(
        self
    ) -> str:
        LOG.info(f"Starting creation of {self.__class__.__name__} agent {self.agent_id}")

        agent_table = await self.db.get_table("agents")
        agent_id = await agent_table.add(self, id=self.agent_id)
        return agent_id

    async def db_save(self) -> str:
        LOG.trace(self.db)
        agent_table = await self.db.get_table("agents")
        agent_id = await agent_table.update(
            agent_id=self.agent_id, user_id=self.user_id, value=self
        )
        return agent_id

    @classmethod
    async def db_list(
        cls,
        user_id: str,
        page: int = 1,
        page_size: int = 10,
    )  -> list[dict] : #-> list[BaseAgent.SystemSettings]:   
        LOG.trace(f"Entering : {cls.__name__}.list_users_agents_from_db()")
        from AFAAS.core.db.table.nosql.agent import AgentsTable
        from AFAAS.interfaces.db.db import AbstractMemory
        from AFAAS.interfaces.db.db_table import AbstractTable

        db_settings = AbstractMemory.SystemSettings()

        db = AbstractMemory.get_adapter(
            db_settings=db_settings
        )
        agent_table: AgentsTable = await db.get_table("agents")

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

        agent_list: list[dict] = await agent_table.list(filter=filter)
        return agent_list

