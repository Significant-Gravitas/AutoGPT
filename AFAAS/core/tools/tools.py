from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from AFAAS.interfaces.agent import BaseAgent
    from AFAAS.interfaces.configuration import *
    from .tool_parameters import ToolParameter

from AFAAS.core.lib.context_items import ContextItem
from AFAAS.core.lib.task import Task
from AFAAS.core.lib.sdk.logger import AFAASLogger
from langchain.tools.base import BaseTool

LOG = AFAASLogger(name=__name__)

ToolReturnValue = Any
ToolOutput = ToolReturnValue | tuple[ToolReturnValue, ContextItem]


class Tool:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    success_check_callback: Callable[..., Any]

    def __init__(
        self,
        name: str,
        description: str,
        exec_function: Callable[..., ToolOutput],
        parameters: list[ToolParameter],
        success_check_callback: Callable[..., Any],
        enabled: Literal[True] | Callable[[Any], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
        available: Literal[True] | Callable[[BaseAgent], bool] = True,
        hide=False,
    ):
        self.name = name
        self.description = description
        self.method = exec_function
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.aliases = aliases
        self.available = available
        self.hide = hide
        self.success_check_callback = success_check_callback

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.method)

    def __call__(self, *args, agent: BaseAgent, **kwargs) -> Any:
        # if callable(self.enabled) and not self.enabled(agent.legacy_config):
        #     if self.disabled_reason:
        #         raise RuntimeError(
        #             f"Tool '{self.name}' is disabled: {self.disabled_reason}"
        #         )
        #     raise RuntimeError(f"Tool '{self.name}' is disabled")

        # if callable(self.available) and not self.available(agent):
        #     raise RuntimeError(f"Tool '{self.name}' is not available")

        return self.method(*args, **kwargs, agent=agent)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.spec.type.value if param.spec.required else f'Optional[{param.spec.type.value}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description.rstrip('.')}. Params: ({', '.join(params)})"

    def default_success_check_callback(self , task: Task, tool_output: Any):
        LOG.trace(f"Tool.default_success_check_callback() called for {self}")
        LOG.debug(f"Task = {task}")
        LOG.debug(f"Tool output = {tool_output}")
        
        return self.description

        #return summary
