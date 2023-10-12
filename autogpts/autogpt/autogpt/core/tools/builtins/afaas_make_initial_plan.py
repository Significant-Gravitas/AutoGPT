"""Tools to control the internal state of the program"""

from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.core.agents.base import BaseAgent

from autogpt.core.agents.base.features.context import get_agent_context
from autogpt.core.utils.exceptions import InvalidArgumentError
from autogpt.core.tools.command_decorator import tool
from autogpt.core.utils.json_schema import JSONSchema

from autogpt.core.agents.simple.lib.models.plan import Plan
from autogpt.core.agents.simple.lib.models.tasks import Task, TaskStatusList

logger = logging.getLogger(__name__)


@tool(
    name = "afaas_make_initial_plan",
    description = "Make a plan to tacle a tasks",
    #parameters = ,
    hide = True
)
async def afaas_make_initial_plan(agent: BaseAgent) -> None:
   # plan =  self.execute_strategy(
    agent._loop.tool_registry().list_tools_descriptions()
    plan = await agent._loop.execute_strategy(
        strategy_name="make_initial_plan",
        agent_name=agent.agent_name,
        agent_role=agent.agent_role,
        agent_goals=agent.agent_goals,
        agent_goal_sentence=agent.agent_goal_sentence,
        description=agent._loop._current_task_routing_description,
        routing_feedbacks=agent._loop._current_task_routing_feedbacks,
        tools=agent._loop.tool_registry().list_tools_descriptions(),
    )

    # TODO: Should probably do a step to evaluate the quality of the generated tasks,
    #  and ensure that they have actionable ready and acceptance criteria

    agent.plan = Plan(
        tasks=[Task.parse_obj(task) for task in plan.parsed_result["task_list"]]
    )
    agent.plan.tasks.sort(key=lambda t: t.priority, reverse=True)
    agent._loop._current_task = agent.plan[-1]
    agent._loop._current_task.context.status = TaskStatusList.READY
    return plan

