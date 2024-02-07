from AFAAS.core.tools.tool import Tool
from AFAAS.lib.task.task import Task


from typing import Any


async def update_agent_goal(
    self: Tool,
    task: Task,
    tool_output: Any
) -> bool:
    # Your implementation here
    task.agent.agent_goal_sentence = task.memory["agent_goal_sentence"]
    task.agent.agent_goals = task.memory["agent_goals"]
    return True
