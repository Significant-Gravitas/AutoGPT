from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from langchain.vectorstores import VectorStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import Field

from AFAAS.configs.schema import Configurable
from AFAAS.core.adapters.openai.chatmodel import AFAASChatOpenAI
from AFAAS.core.workspace.local import AGPTLocalFileWorkspace
from AFAAS.interfaces.adapters import AbstractLanguageModelProvider
from AFAAS.interfaces.adapters.embeddings.wrapper import (
    ChromaWrapper,
    VectorStoreWrapper,
)
from AFAAS.core.agents.prompt_manager import BasePromptManager
from AFAAS.interfaces.agent.loop import BaseLoopHook
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.db.db import AbstractMemory
from AFAAS.interfaces.workflow import BaseWorkflow, WorkflowRegistry
from AFAAS.interfaces.workspace import AbstractFileWorkspace
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task

from .loop import RoutingLoop

LOG = AFAASLogger(name=__name__)


class RoutingAgent(BaseAgent, Configurable):
    class SystemSettings(BaseAgent.SystemSettings):
        agent_name: str = Field(default="RoutingAgent")
        parent_agent_id: str
        parent_agent: BaseAgent
        current_task: Task
        note_to_agent_length: int

        def dict_db(self, **dumps_kwargs: Any) -> dict:
            return super().dict_db(**dumps_kwargs)

        class Config(BaseAgent.SystemSettings.Config):
            pass

    def __init__(
        self,
        settings: RoutingAgent.SystemSettings,
        user_id: str,
        agent_id: str = SystemSettings.generate_uuid(),
        loop: RoutingLoop = RoutingLoop(),
        prompt_manager: BasePromptManager = BasePromptManager(),
        db: AbstractMemory = None,
        default_llm_provider: AbstractLanguageModelProvider = None,
        workspace: AbstractFileWorkspace = None,
        vectorstore: VectorStoreWrapper = None,
        embeddings: Embeddings = None,
        workflow_registry: WorkflowRegistry = None,
        log_path=None,
        **kwargs,
    ):
        super().__init__(
            settings=settings,
            db=db,
            workspace=workspace,
            default_llm_provider=default_llm_provider,
            prompt_manager=prompt_manager,
            user_id=user_id,
            agent_id=agent_id,
            vectorstore=vectorstore,
            embedding_model=embeddings,
            workflow_registry=workflow_registry,
            log_path=log_path,
            **kwargs,
        )

        self.parrent_agent: BaseAgent = settings.parent_agent
        self.parrent_agent_id: str = settings.parent_agent_id
        self.current_task = settings.current_task

        self._loop: RoutingLoop = loop
        self._loop.set_agent(agent=self)

    def loophooks(self) -> RoutingLoop.LoophooksDict:
        if not self._loop._loophooks:
            self._loop._loophooks = {}
        return self._loop._loophooks

    def loop(self) -> RoutingLoop:
        return self._loop

    ################################################################################
    ################################ LOOP MANAGEMENT################################
    ################################################################################

    async def start(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        return_var = await super().start(
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    async def stop(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        return_var = await super().stop(
            agent=self,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    ################################################################################
    ################################FACTORY SPECIFIC################################
    ################################################################################

    # @classmethod
    # def get_strategies(cls)-> list:
    #     from AFAAS.core.agents.routing.strategies import \
    #         StrategiesSet
    #     return StrategiesSet.get_strategies()
    def __repr__(self):
        return "RoutingAgent()"
