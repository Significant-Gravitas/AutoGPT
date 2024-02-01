from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.lib.task.plan import Plan
    from AFAAS.interfaces.db.db_table import AbstractTable
    from AFAAS.interfaces.adapters import \
        CompletionModelFunction , AbstractChatModelResponse
    from AFAAS.interfaces.tools.base import AbstractToolRegistry
    from AFAAS.core.tools.tool import Tool
    from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy

    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name="autogpt")

class AgentMixin:
    _agent: BaseAgent

    def __init__(self, **kwargs):
        self._agent = None

    def set_agent(self, agent: BaseAgent):
        if hasattr(self, "_agent") and self._agent is not None:
            LOG.notice(f"Agent already set")
        else :
            LOG.info(f"Setting agent {agent.agent_id} for {self.__class__.__name__}")

        self._agent = agent

    ###
    ## Save agent component
    ###
    async def save_agent(self):
        return await self._agent.db_save()

    async def save_plan(self):
        return await self._agent.plan.db_save()

    ###
    ## Messaging
    ###
    def message_user(self, message: str):
        LOG.warning("Deprecated , use user_interaction")
        return self._agent._user_input_handler(message)

    def get_user_input(self, message: str):
        LOG.warning("Deprecated , use user_interaction")
        return self._agent._user_input_handler(message)

    ###
    ## Shorcuts
    ###
    def tool_registry(self) -> AbstractToolRegistry:
        return self._agent.tool_registry

    def get_tool_list(self) -> list[Tool]:
        return self.tool_registry().get_tool_list()

    def get_tools_as_functions_for_api(self) -> list[CompletionModelFunction]:
        self.tool_registry().dump_tools()

    def plan(self) -> Plan:
        return self._agent.plan

    async def get_table(self, table_name: str) -> AbstractTable:
        return await self._agent.db.get_table(table_name=table_name)

    def get_strategy(self, strategy_name: str) -> AbstractPromptStrategy:
        return self._agent._prompt_manager.get_strategy(strategy_name=strategy_name)

    async def _execute_strategy(
        self, strategy_name: str, **kwargs
    ) -> AbstractChatModelResponse:
        return await self._agent._prompt_manager.execute_strategy(
            strategy_name=strategy_name, **kwargs
        )
