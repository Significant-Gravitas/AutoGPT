"""Tools to control the internal state of the program"""

from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.app.core.agents.base import BaseAgent

from AFAAS.app.lib.task.task import Task
from AFAAS.app.sdk import forge_log
from AFAAS.app.core.agents.routing import RoutingAgent
from AFAAS.app.core.agents.routing.strategies.routing import \
    RoutingStrategyConfiguration
from AFAAS.app.core.tools.command_decorator import tool

logger = forge_log.ForgeLogger(__name__)


@tool(
    name="afaas_routing",
    description="Assist user refining it's requirements thus improving LLM responses",
    # parameters = ,
    hide=True,
)
async def afaas_routing(
    task: Task,
    agent: BaseAgent,
    note_to_agent_length: int = RoutingStrategyConfiguration().note_to_agent_length,
) -> None:
    """
    Tool that help an agent to decide what kind of planning / execution to undertake
    """
    try:
        # USER CONTEXT AGENT : Create Agent Settings
        routing_settings: RoutingAgent.SystemSettings = RoutingAgent.SystemSettings(
            user_id=agent.user_id,
            parent_agent_id=agent.agent_id,
            parent_agent=agent,
            current_task=task,
            note_to_agent_length=note_to_agent_length,
        )
        # routing_settings.agent_goals=  agent.agent_goals
        # routing_settings.agent_goal_sentence=  agent.agent_goal_sentence
        routing_settings.memory = agent._memory._settings
        routing_settings.workspace = agent._workspace._settings
        routing_settings.chat_model_provider = agent._chat_model_provider._settings
        routing_settings.note_to_agent_length = note_to_agent_length

        # USER CONTEXT AGENT : Save RoutingAgent Settings in DB (for POW / POC)
        new_routing_agent: RoutingAgent = RoutingAgent.create_agent(
            agent_settings=routing_settings, logger=agent._logger
        )

        # # USER CONTEXT AGENT : Get RoutingAgent from DB (for POW / POC)
        routing_settings.agent_id = new_routing_agent.agent_id

        routing_agent: RoutingAgent = RoutingAgent.get_instance_from_settings(
            agent_settings=routing_settings,
            logger=agent._logger,
        )

        routing_return: dict = await routing_agent.run(
            user_input_handler=agent._user_input_handler,
            user_message_handler=agent._user_message_handler,
        )

        # agent.agent_goal_sentence =     routing_return["agent_goal_sentence"]
        # agent.agent_goals =     routing_return["agent_goals"]

        return routing_return
    except Exception as e:
        raise str(e)
