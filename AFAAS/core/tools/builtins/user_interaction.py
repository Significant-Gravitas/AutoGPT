"""Tools to interact with the user"""

from __future__ import annotations

TOOL_CATEGORY = "user_interaction"
TOOL_CATEGORY_TITLE = "User Interaction"

# from AFAAS.lib.app import clean_input
from AFAAS.core.tools.tool_decorator import tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema


@tool(
    "user_interaction",
    (
        "If you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    {
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The question or prompt to the user",
            required=True,
        )
    },
    enabled=lambda config: not config.noninteractive_mode,
)
async def user_interaction(question: str, task: Task, agent: BaseAgent) -> str:
    pass

    # resp = await clean_input(
    #     agent.legacy_config, f"{agent.ai_config.ai_name} asks: '{question}': "
    # )
    # TODO : MAke user-proxy here
    # TODO : Save Agent output
    user_response = agent._user_input_handler(question)
    # TODO : Save User Input
    return user_response
