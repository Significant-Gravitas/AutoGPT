from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.tools.builtins.query_language_model import query_language_model
from AFAAS.core.tools.builtins.user_interaction import user_interaction
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.adapters import AbstractChatModelResponse
from AFAAS.interfaces.tools.tool import AFAASBaseTool
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


@tool(
    name="search_info",
    description=(
        "Find a publicly available information, will find information availale on the web, if fails the user will provide you with the information you seek."
    ),
    parameters={
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A query for a language model. A query should contain a question and any relevant context.",
            required=True,
        ),
        "reasoning": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Detail the process of though that lead you to write this query ",
            required=True,
        ),
    },
    categories=[AFAASBaseTool.FRAMEWORK_CATEGORY],
)
async def search_info(query: str, reasoning: str, task: Task, agent: BaseAgent) -> str:
    search_result: AbstractChatModelResponse = await agent.execute_strategy(
        strategy_name="search_info",
        agent=agent,
        task=task,
        query=query,
        reasoning=reasoning,
        tools=[agent.tool_registry.get_tool("query_language_model")],
    )

    command_name = search_result.parsed_result[0]["command_name"]
    command_args = search_result.parsed_result[0]["command_args"]
    assistant_reply_dict = search_result.parsed_result[0]["assistant_reply_dict"]

    # NOTE: search info MVP (V1) only search info via the LLM
    # TODO: search info V2 will follow this process :
    #    search_info via LLM
    #    analyse output, if output the  search news or   info via the LLM and the web
    if command_name == "query_language_model":
        result = await query_language_model(task=task, agent=agent, **command_args)

    # V2 : We check success here

    if True:  # V1
        return result
    else:
        # V2
        if command_name == "user_interaction":
            raise NotImplementedError("Command user_interaction not implemented")
            return await user_interaction(task=task, agent=agent, **command_args)
        elif command_name == "web_search":
            try:
                from AFAAS.core.tools.untested.web_search import web_search

                return await web_search(task=task, agent=agent, **command_args)
            except:
                raise NotImplementedError("Command search_web not implemented")
        else:
            raise NotImplementedError(f"Command {command_name} not implemented")
