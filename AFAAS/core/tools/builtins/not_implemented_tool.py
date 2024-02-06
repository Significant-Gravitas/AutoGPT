from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.interfaces.tools.tool import AFAASBaseTool
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


async def not_implemented_tool(task: Task, agent: BaseAgent, **kwargs) -> None:
    from AFAAS.core.tools.builtins.user_interaction import user_interaction
    from AFAAS.core.tools.tool_decorator import tool  # Dynamic import

    @tool(
        name="not_implemented_tool",
        description="Ask a user to perform a task that is currently not supported by the system.",
        categories=[AFAASBaseTool.FRAMEWORK_CATEGORY],
    )
    async def inner_not_implemented_tool(
        task: Task, agent: BaseAgent, **kwargs
    ) -> None:
        # Function implementation
        new_query = kwargs.pop("query", "") + "\n"
        if kwargs:
            new_query += "The following parameters were provided:\n" + "".join(
                f" - {k}: {v}\n" for k, v in kwargs.items()
            )
        return await user_interaction(
            query=new_query, task=task, agent=agent, skip_proxy=True
        )

    return await inner_not_implemented_tool(task, agent, **kwargs)
