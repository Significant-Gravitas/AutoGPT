"""Commands to search the web with"""

from __future__ import annotations

from forge.sdk.model import TaskRequestBody 
from autogpt.command_decorator import command
from autogpt.agents.agent_member import AgentMember
from autogpt.core.utils.json_schema import JSONSchema

COMMAND_CATEGORY = "request_agent"
COMMAND_CATEGORY_TITLE = "Request an agent"

@command(
    "request_agent",
    "Request a new agent member for someone. The prompt for this step should be create someone to do this task.",
    {
        "prompt": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The description for agent that one to be created",
            required=True,
        ),
        "role": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Role of agent member that one be created",
            required=True,
        ),
        "boss_id": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The agent id that is going to be the boss of new agent",
            required=True,
        ),
    },
)
async def request_agent(prompt: str, role:str, agent: AgentMember, boss_id: str) -> str:
    """Request new agent for some one

    Args:
        prompt (str): The description for agent that one to be created.
        role (str): role of agent member that one be created.
        boss_id (str): The agent id that is going to be the boss of new agent.

    """
    try:
        if agent.recruiter != None:
            await agent.recruiter.create_task(task_request=TaskRequestBody(input=f"hire someone with {role} and this prompt: {prompt} for agent with id {boss_id}"))
            return f"create task for recruiter to hire {role}"
        elif agent.boss != None:
            await agent.boss.create_task(task_request=TaskRequestBody(input=f"hire someone with {role} and this prompt: {prompt} for agent with id {boss_id}"))
            return f"create task for boss to hire {role}"
        else:
            raise Exception("We can't hire someone ")
    except Exception as ex:
        print(ex)
        return f"can't create {role}"

