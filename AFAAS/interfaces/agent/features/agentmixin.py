from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.lib.task.plan import Plan
    from AFAAS.interfaces.db.db_table import AbstractTable
    from AFAAS.interfaces.adapters import \
        CompletionModelFunction , AbstractChatModelResponse
    from AFAAS.interfaces.tools.base import BaseToolsRegistry
    from AFAAS.core.tools.tools import Tool
    from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy

    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

class AgentMixin:
    _agent: BaseAgent

    def __init__(self, **kwargs):
        self._agent = None

    def set_agent(self, agent: BaseAgent):
        if hasattr(self, "_agent") and self._agent is not None:
            LOG.warning(f"Agent already set")
        else :
            LOG.info(f"Setting agent {agent.agent_id} for {self.__class__.__name__}")

        self._agent = agent

    ###
    ## Save agent component
    ###
    async def save_agent(self):
        return self._agent.save_agent_in_memory()

    async def save_plan(self):
        return await self._agent.plan.save()

    ###
    ## Messaging
    ###
    def message_user(self, message: str):
        return self._agent._user_input_handler(message)

    def get_user_input(self, message: str):
        return self._agent._user_input_handler(message)

    ###
    ## Shorcuts
    ###
    def tool_registry(self) -> BaseToolsRegistry:
        return self._agent._tool_registry

    def get_tool_list(self) -> list[Tool]:
        return self.tool_registry().get_tool_list()

    def get_tools_as_functions_for_api(self) -> list[CompletionModelFunction]:
        self.tool_registry().dump_tools()

    def plan(self) -> Plan:
        return self._agent.plan

    def get_table(self, table_name: str) -> AbstractTable:
        return self._agent.memory.get_table(table_name=table_name)

    def get_strategy(self, strategy_name: str) -> AbstractPromptStrategy:
        return self._agent._prompt_manager._prompt_strategies[strategy_name]

    async def _execute_strategy(
        self, strategy_name: str, **kwargs
    ) -> AbstractChatModelResponse:
        return await self._agent._prompt_manager.execute_strategy(
            strategy_name=strategy_name, **kwargs
        )
