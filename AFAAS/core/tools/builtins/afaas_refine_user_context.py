"""Tools to control the internal state of the program"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.agents.usercontext.main import UserContextAgent
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.tools.base import AbstractTool
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task

LOG = AFAASLogger(name=__name__)


@tool(
    name="afaas_refine_user_context",
    description="Assist user refining it's requirements thus improving LLM responses",
    # parameters = ,
    hide=True,
    categories=["famework"],
)
async def afaas_refine_user_context(task: Task, agent: BaseAgent) -> None:
    """
    Configures the user context agent based on the current agent settings and executes the user context agent.
    Returns the updated agent goals.
    """
    try:
        # USER CONTEXT AGENT : Create Agent Settings
        usercontext_settings: UserContextAgent.SystemSettings = (
            UserContextAgent.SystemSettings(
                user_id=agent.user_id,
                parent_agent_id=agent.agent_id,
                parent_agent=agent,
            )
        )

        # FIXME: Define wich dependency to inject
        user_context_agent = UserContextAgent(
            settings=usercontext_settings,
            **usercontext_settings.dict(),
            task = task
        )
        # NOTE: We don't save the agent
        # new_user_context_agent = await UserContextAgent.create_agent()

        user_context_return: dict = await user_context_agent.run(
            user_input_handler=agent._user_input_handler,
            user_message_handler=agent._user_message_handler,
        )

        # FIXME:0.0.2: Move it outside of the tool
        agent.agent_goal_sentence = user_context_return["agent_goal_sentence"]
        agent.agent_goals = user_context_return["agent_goals"]
    except Exception as e:
        raise str(e)
