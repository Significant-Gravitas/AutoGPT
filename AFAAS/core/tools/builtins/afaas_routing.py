from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.agents.routing.pipeline import RoutingPipeline
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.tools.tool import AFAASBaseTool
from AFAAS.prompts.routing import RoutingStrategyConfiguration



from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger


from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)


@tool(
    name="afaas_routing",
    description="Divide a task into subtasks",
    tech_description="Divide a task into subtasks",
    # parameters = ,
    hide=True,
    categories=[AFAASBaseTool.FRAMEWORK_CATEGORY, "planning"],
)
async def afaas_routing(
    task: AbstractTask,
    agent: BaseAgent,
    note_to_agent_length: int = RoutingStrategyConfiguration().note_to_agent_length,
) -> None:
    """
    Tool that help an agent to decide what kind of planning / execution to undertake
    """
    try:
        pipeline = RoutingPipeline(task=task, agent=agent, note_to_agent_length = note_to_agent_length) 
        return await pipeline.execute()
    except Exception as e:
        raise str(e)
