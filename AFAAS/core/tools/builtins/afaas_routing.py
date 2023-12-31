"""Tools to control the internal state of the program"""

from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.agents.routing.main import RoutingAgent
from AFAAS.core.tools.tool_decorator import tool
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.prompts.routing import RoutingStrategyConfiguration

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

        #FIXME: Define wich dependency to inject
        routing_agent = RoutingAgent(
            settings = routing_settings,
            **RoutingStrategyConfiguration().dict(),
        )
        #NOTE: We don't save the agent
        #new_user_context_agent = UserContextAgent.create_agent()

        routing_return: dict = await routing_agent.run(
            user_input_handler=agent._user_input_handler,
            user_message_handler=agent._user_message_handler,
        )

        return routing_return
    except Exception as e:
        raise str(e)
