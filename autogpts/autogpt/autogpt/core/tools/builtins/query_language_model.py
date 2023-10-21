from __future__ import annotations

TOOL_CATEGORY = "framework"
TOOL_CATEGORY_TITLE = "Framework"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.agents.base import BaseAgent

from autogpts.autogpt.autogpt.core.agents.base.features.context import get_agent_context
from autogpts.autogpt.autogpt.core.utils.exceptions import InvalidArgumentError
from autogpts.autogpt.autogpt.core.tools.command_decorator import tool
from autogpts.autogpt.autogpt.core.utils.json_schema import JSONSchema


logger = logging.getLogger(__name__)


@tool(
    name = "query_language_model",
    description = (
        "Search the web and with the capacity of returning steerable & structured result. Not very good at retriving data published over the last 2 years."
    ),
    parameters = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A query for a language model. A query should contain a question and any relevant context.",
            required=True
        ),
        "format": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Describe the format (plan, length,...) of the expected answer.",
        ),
        "answer_as": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Describe person with his profession, mindeset, personnality ect... You would like to get an answer from.",
        )
    }
)
async def query_language_model(agent: BaseAgent) -> None:
   # plan =  self.execute_strategy(
    agent._loop.tool_registry().list_tools_descriptions()
    plan = await agent._loop.execute_strategy(
        strategy_name="make_initial_plan",
        agent=agent
    )

    return plan

