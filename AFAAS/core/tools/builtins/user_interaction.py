"""Tools to interact with the user"""

from __future__ import annotations

# from AFAAS.lib.app import clean_input
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.tools.base import AbstractTool
from AFAAS.lib.message_agent_user import Emiter, MessageAgentUser
from AFAAS.lib.message_common import AFAASMessageStack
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema


@tool(
    name="user_interaction",
    description=(
        "Ask a question to the user if you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    parameters={
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The question or prompt to the user",
            required=True,
        )
    },
    categories=[AbstractTool.FRAMEWORK_CATEGORY],
)
async def user_interaction(
    query: str, task: Task, agent: BaseAgent, skip_proxy=False
) -> str:
    if skip_proxy:
        if False:  # TODO: Make user-proxy here
            pass
        if False and True:  # If the user proxy found an answer
            return await agent._user_input_handler(query)

        # TODO: Create  message but as "hidden"

    await agent.message_agent_user.db_create(
        message=MessageAgentUser(
            emitter=Emiter.AGENT.value,
            user_id=agent.user_id,
            agent_id=agent.agent_id,
            message=str(query),
        )
    )
    user_response = await agent._user_input_handler(query)

    await agent.message_agent_user.db_create(
        message=MessageAgentUser(
            emitter=Emiter.USER.value,
            user_id=agent.user_id,
            agent_id=agent.agent_id,
            message=str(user_response),
        )
    )
    return user_response
