from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.core.agents.base import BaseAgent

from AFAAS.core.lib.task.task import Task
from AFAAS.core.lib.sdk.logger import AFAASLogger
from AFAAS.core.tools.command_decorator import tool
from AFAAS.core.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


@tool(
    name="query_language_model",
    description=(
        "Search the web and with the capacity of returning steerable & structured result. Not very good at retriving data published over the last 2 years."
    ),
    parameters={
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A query for a language model. A query should contain a question and any relevant context.",
            required=True,
        ),
        "format": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Describe the format (plan, length,...) of the expected answer.",
        ),
        "answer_as": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Describe person with his profession, mindeset, personnality ect... You would like to get an answer from.",
        ),
    },
)
async def query_language_model(task: Task, agent: BaseAgent) -> None:
    # plan =  self.execute_strategy(
    agent._loop.tool_registry().list_tools_descriptions()
    plan = await agent._loop._execute_strategy(
        strategy_name="make_initial_plan", agent=agent
    )

    return plan
