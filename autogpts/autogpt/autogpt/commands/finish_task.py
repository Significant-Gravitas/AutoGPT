"""Commands to control the internal state of the program"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogpt.command_decorator import command
from autogpt.agents.utils.exceptions import TaskFinished
from autogpt.core.utils.json_schema import JSONSchema

COMMAND_CATEGORY = "finish"
COMMAND_CATEGORY_TITLE = "Finish"


if TYPE_CHECKING:
    from autogpt.agents.agent import Agent


logger = logging.getLogger(__name__)


@command(
    "finish_task",
    "Use this to shut down once you have completed your task,"
    " or when there are insurmountable problems that make it impossible"
    " for you to finish your task.",
    {
        "reason": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="A summary to the user of how the goals were accomplished",
            required=True,
        ),
        "task_id": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The task id that is gone",
            required=True,
        ),
    },
)
def finish(reason: str, task_id: str, agent: Agent) -> None:
    """
    A function that finish a task

    Parameters:
        reason (str): A summary to the user of how the goals were accomplished.
        task_id (str): The task id that is gone.
    Returns:
        A result string from create chat completion. A list of suggestions to
            improve the code.
    """
    raise TaskFinished(task_id, reason)
