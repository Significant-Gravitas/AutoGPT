"""Tools to control the internal state of the program"""

from __future__ import annotations

TOOL_CATEGORY = "system"
TOOL_CATEGORY_TITLE = "System"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.core.agents.base import BaseAgent

from AFAAS.core.lib.task.task import Task
from AFAAS.core.lib.sdk.logger import AFAASLogger
from AFAAS.core.agents.base.features.context import \
    get_agent_context
from AFAAS.core.tools.command_decorator import tool
# from AFAAS.core.lib.sdk.errors import AgentTerminated, InvalidArgumentError
from AFAAS.core.lib.sdk.errors import InvalidArgumentError
from AFAAS.core.utils.json_schema import JSONSchema

logger = AFAASLogger(name=__name__)


@tool(
    "finish",
    "Use this to shut down once you have completed your task,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    {
        "reason": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A summary to the user of how the goals were accomplished",
            required=True,
        )
    },
)
def finish(reason: str, task: Task, agent: BaseAgent) -> None:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    raise NotImplementedError
    raise AgentTerminated(reason)


@tool(
    "hide_context_item",
    "Hide an open file, folder or other context item, to save memory.",
    {
        "number": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The 1-based index of the context item to hide",
            required=True,
        )
    },
    available=lambda a: bool(get_agent_context(a)),
)
def close_context_item(number: int, task: Task, agent: BaseAgent) -> str:
    assert (context := get_agent_context(agent)) is not None

    if number > len(context.items) or number == 0:
        raise InvalidArgumentError(f"Index {number} out of range")

    context.close(number)
    return f"Context item {number} hidden ✅"
