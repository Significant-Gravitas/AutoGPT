from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.tools.tool import AFAASBaseTool
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


@tool(
    name="query_language_model",
    description=(
        "Answer any question with the capacity of returning steerable & structured result. Limitation : Can not use data published last month. Tips: The more informations you give the better the answer will be."
    ),
    parameters={
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A query for a language model. A query should contain a question and any relevant context.",
            required=True,
        ),
        "format": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Describe the ideal format (plan, length,...) for the expected answer.",
            required=True,
        ),
        "persona": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Assertive description of the ideal persona with his profession, expertise, mindset, factuality, personnality, ect... You would like to get an answer from.",
            required=True,
        ),
        "example": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Optional : If given, will accurately will return a result reproducing the patern of the example",
        ),
    },
    categories=[AFAASBaseTool.FRAMEWORK_CATEGORY],
)
async def query_language_model(
    task: Task,
    agent: BaseAgent,
    query: str,
    format: str,
    persona: str,
    example: str = None,
) -> None:
    response = await agent.execute_strategy(
        strategy_name="query_llm",
        agent=agent,
        task=task,
        query=query,
        format=format,
        persona=persona,
        example=example,
    )

    return response.parsed_result
