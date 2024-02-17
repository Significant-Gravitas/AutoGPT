"""Commands to search the web with"""

from __future__ import annotations

from autogpt.command_decorator import command
from autogpt.agents.agent_member import AgentMember
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.agents.agent_group import create_agent_member

COMMAND_CATEGORY = "create_agent"
COMMAND_CATEGORY_TITLE = "Create agent"


@command(
    "create_agent",
    "Create a new agent member for someone. The prompt for this step should be create someone to do this task.",
    {
        "prompt": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The description for agent that one to be created",
            required=True,
        ),
        "role": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="role of agent member that one be created",
            required=True,
        ),
        "boss_id": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The agent who will be boss of new agent id",
            required=True,
        ),
    },
)
async def create_agent(prompt: str, role: str, agent: AgentMember, boss_id: str) -> str:
    """Create new agent for some one

    Args:
        prompt (str): The description for agent that one to be created.
        role (str): role of agent member that one be created.

    """
    try:
        group = agent.group
        boss = group.members[boss_id]
        await create_agent_member(
            role=role, initial_prompt=prompt, boss=boss, llm_provider=agent.llm_provider
        )
        return f"{role} created"
    except Exception as ex:
        print(ex)
        return f"can't create {role}"
