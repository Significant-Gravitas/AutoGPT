"""Commands to control the internal state of the program"""

from __future__ import annotations

COMMAND_CATEGORY = "system"
COMMAND_CATEGORY_TITLE = "System"

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

from autogpt.agents.features.context import get_agent_context
from autogpt.agents.utils.exceptions import InvalidArgumentError
from autogpt.command_decorator import command

logger = logging.getLogger(__name__)


@command(
    "finish",
    "Use this to shut down once you have accomplished all of your goals,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    {
        "reason": {
            "type": "string",
            "description": "A summary to the user of how the goals were accomplished",
            "required": True,
        }
    },
)
def finish(reason: str, agent: Agent) -> None:
    """
    A function that takes in a string and exits the program

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    logger.info(reason, extra={"title": "Shutting down...\n"})
    quit()


@command(
    "hide_context_item",
    "Hide an open file, folder or other context item, to save memory.",
    {
        "number": {
            "type": "integer",
            "description": "The 1-based index of the context item to hide",
            "required": True,
        }
    },
    available=lambda a: bool(get_agent_context(a)),
)
def close_context_item(number: int, agent: Agent) -> str:
    assert (context := get_agent_context(agent)) is not None

    if number > len(context.items) or number == 0:
        raise InvalidArgumentError(f"Index {number} out of range")

    context.close(number)
    return f"Context item {number} hidden ✅"
