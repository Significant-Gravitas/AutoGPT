"""Tools to control the internal state of the program"""

from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent import BaseAgent

from AFAAS.lib.task.task import Task
from AFAAS.core.agents.usercontext import UserContextAgent
from AFAAS.core.tools.command_decorator import tool

from AFAAS.lib.sdk.logger import AFAASLogger
LOG =  AFAASLogger(name=__name__)


@tool(
    name="afaas_refine_user_context",
    description="Assist user refining it's requirements thus improving LLM responses",
    # parameters = ,
    hide=True,
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
        usercontext_settings.agent_goals = agent.agent_goals
        usercontext_settings.agent_goal_sentence = agent.agent_goal_sentence
        usercontext_settings.memory = agent._memory._settings
        usercontext_settings.workspace = agent._workspace._settings
        usercontext_settings.chat_model_provider = agent._chat_model_provider._settings

        # FIXME: REMOVE WHEN WE GO LIVE
        # USER CONTEXT AGENT : Save UserContextAgent Settings in DB (for POW / POC)
        new_user_context_agent = UserContextAgent.create_agent(
            agent_settings=usercontext_settings
        )
        # USER CONTEXT AGENT : Get UserContextAgent from DB (for POW / POC)
        usercontext_settings.agent_id = new_user_context_agent.agent_id

        user_context_agent = UserContextAgent.get_instance_from_settings(
            agent_settings=usercontext_settings,
        )

        user_context_return: dict = await user_context_agent.run(
            user_input_handler=agent._user_input_handler,
            user_message_handler=agent._user_message_handler,
        )

        agent.agent_goal_sentence = user_context_return["agent_goal_sentence"]
        agent.agent_goals = user_context_return["agent_goals"]
    except Exception as e:
        raise str(e)
