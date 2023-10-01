"""Commands to interact with the user"""

from __future__ import annotations

COMMAND_CATEGORY = "user_interaction"
COMMAND_CATEGORY_TITLE = "User Interaction"

from autogpt.core.agents.base import BaseAgent
#from autogpt.core.utils.app import clean_input
from autogpt.core.tools.command_decorator  import tool
from autogpt.core.utils.json_schema import JSONSchema


@tool(
    "ask_user",
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
async def ask_user(question: str, agent: BaseAgent) -> str:
    # resp = await clean_input(
    #     agent.legacy_config, f"{agent.ai_config.ai_name} asks: '{question}': "
    # )
    return agent._user_input_handler(question)
