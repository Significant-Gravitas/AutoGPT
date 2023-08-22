"""Commands to control the internal state of the program"""

from __future__ import annotations

COMMAND_CATEGORY = "system"
COMMAND_CATEGORY_TITLE = "System"

import logging

from autogpt.agents.agent import Agent
from autogpt.command_decorator import command

logger = logging.getLogger(__name__)


@command(
    "goals_accomplished",
    "Goals are accomplished and there is nothing left to do",
    {
        "reason": {
            "type": "string",
            "description": "A summary to the user of how the goals were accomplished",
            "required": True,
        }
    },
)
def task_complete(reason: str, agent: Agent) -> None:
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
