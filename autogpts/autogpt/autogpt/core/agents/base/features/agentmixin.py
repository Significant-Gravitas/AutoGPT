from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..main import BaseAgent
    from autogpt.core.tools.base import BaseToolsRegistry 

    from autogpt.core.tools.tools import Tool 

class AgentMixin:

    _agent : BaseAgent 
    def __init__(self, **kwargs) :
        pass

    def set_agent(self, agent : BaseAgent) :
        self._agent = agent

    def get_tools(self) -> list[Tool] :
        return self._agent._tool_registry.get_tools()
    
    def tool_registry(self) -> BaseToolsRegistry :
        return self._agent._tool_registry
    
    async def save_agent(self) : 
        return self._agent.save_agent_in_memory()
