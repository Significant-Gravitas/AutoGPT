from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.core.lib.task.plan import Plan
    from AFAAS.core.memory.table import AbstractTable
    from AFAAS.core.resource.model_providers import \
        CompletionModelFunction , ChatModelResponse
    from AFAAS.core.tools.base import BaseToolsRegistry
    from AFAAS.core.tools.tools import Tool
    from AFAAS.core.prompting.base import BasePromptStrategy

    from ..main import BaseAgent


class AgentMixin:
    _agent: BaseAgent

    def __init__(self, **kwargs):
        pass

    def set_agent(self, agent: BaseAgent):
        if hasattr(self, "_agent") and self._agent is not None:
            raise Exception("Agent already set")

        self._agent = agent

    ###
    ## Save agent component
    ###
    async def save_agent(self):
        return self._agent.save_agent_in_memory()

    async def save_plan(self):
        return self._agent.plan.save()

    ###
    ## Messaging
    ###
    def message_user(self, message: str):
        return self._agent._user_input_handler(message)

    def get_user_input(self, message: str):
        return self._agent._user_input_handler(message)

    def logger(self) -> Logger:
        return self._agent._logger

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
        return self._agent._memory.get_table(table_name=table_name)

    def get_strategy(self, strategy_name: str) -> BasePromptStrategy:
        return self._agent._prompt_manager._prompt_strategies[strategy_name]

    async def _execute_strategy(
        self, strategy_name: str, **kwargs
    ) -> ChatModelResponse:
        return await self._agent._prompt_manager.execute_strategy(
            strategy_name=strategy_name, **kwargs
        )
