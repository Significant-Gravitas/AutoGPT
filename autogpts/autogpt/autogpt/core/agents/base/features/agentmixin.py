from __future__ import annotations

from typing import TYPE_CHECKING, Any
from logging import Logger

if TYPE_CHECKING:
    from ..main import BaseAgent
    from autogpt.core.tools.base import BaseToolsRegistry 

    from autogpt.core.tools.tools import Tool 

class AgentMixin:

    _agent : BaseAgent 
    def __init__(self, **kwargs) :
        pass

    def set_agent(self, agent : BaseAgent) :
        if hasattr(self, '_agent') and self._agent is not None:
            raise Exception("Agent already set")
         
        self._agent = agent

    ###
    ## Save agent component
    ###
    async def save_agent(self) : 
        return self._agent.save_agent_in_memory()
    
    async def save_plan(self) : 
        return self._agent.plan.save()
    
    ###
    ## Messaging
    ###
    def message_user(self , message : str) : 
        return self._agent._user_input_handler(message)
    
    def get_user_input(self , message : str) : 
        return self._agent._user_input_handler(message)
    
    def logger(self) -> Logger : 
        return self._agent._logger
    
    ###
    ## Shorcutes
    ###
    def tool_registry(self) -> BaseToolsRegistry :
        return self._agent._tool_registry
    
    def get_tools(self) -> list[Tool] :
        return self._agent._tool_registry.get_tools()
    
    def plan(self)  : 
        return self._agent.plan
    
    
