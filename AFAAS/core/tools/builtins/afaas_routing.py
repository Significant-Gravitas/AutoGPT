"""Tools to control the internal state of the program"""

from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent import BaseAgent

from AFAAS.lib.task.task import Task
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.core.agents.routing import RoutingAgent
from AFAAS.prompts.routing import RoutingStrategyConfiguration
from AFAAS.core.tools.command_decorator import tool

LOG = AFAASLogger(name=__name__)


@tool(
    name="afaas_routing",
    description="Divide a task into subtasks",
    tech_description="Divide a task into subtasks",
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
        routing_settings.memory = agent.memory._settings
        routing_settings.note_to_agent_length = note_to_agent_length

        # USER CONTEXT AGENT : Save RoutingAgent Settings in DB (for POW / POC)
        new_routing_agent: RoutingAgent = RoutingAgent.create_agent(
            agent_settings=routing_settings,
            workspace=agent._workspace,
            default_llm_provider=agent.default_llm_provider,
        )

        # # USER CONTEXT AGENT : Get RoutingAgent from DB (for POW / POC)
        routing_settings.agent_id = new_routing_agent.agent_id

        routing_agent: RoutingAgent = RoutingAgent.get_instance_from_settings(
            agent_settings=routing_settings,
            workspace=agent._workspace,
            default_llm_provider=agent.default_llm_provider,
        )

        routing_return: dict = await routing_agent.run(
            user_input_handler=agent._user_input_handler,
            user_message_handler=agent._user_message_handler,
        )

        return routing_return
    except Exception as e:
        raise str(e)
