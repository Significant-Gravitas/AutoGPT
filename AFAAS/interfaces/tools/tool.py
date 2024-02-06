from __future__ import annotations
from AFAAS.interfaces.adapters import CompletionModelFunction
from AFAAS.interfaces.tools.tool_output import ToolOutput
from AFAAS.interfaces.tools.tool_parameters import ToolParameter


from langchain.tools.base import BaseTool


from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.task.task import AbstractTask


from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional


class AFAASBaseTool(ABC):
    """An abstract class representing a tool."""

    FRAMEWORK_CATEGORY = "framework"


    @abstractmethod
    def __call__(self, *args, agent: BaseAgent, **kwargs) -> Any:
        pass

    @abstractmethod
    def dump(self) -> CompletionModelFunction:
        pass

    @classmethod
    @abstractmethod
    def generate_from_langchain_tool(
        cls,
        langchain_tool: BaseTool,
        arg_converter: Optional[Callable] = None,
        categories: list[str] = ["undefined"],
        success_check_callback=None,
        make_summarry_function=None,
    ) -> AFAASBaseTool:
        pass

    @abstractmethod
    async def default_tool_success_check_callback(
        self, task: AbstractTask, tool_output: Any
    ):
        pass

    @abstractmethod
    async def default_tool_execution_summarry(
        self, task: AbstractTask, tool_output: Any
    ):
        pass
