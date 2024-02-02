from __future__ import annotations

import inspect
import asyncio
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from AFAAS.interfaces.tools.tool_output import ToolOutput
from AFAAS.interfaces.tools.tool_parameters import ToolParameter
from AFAAS.lib.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from langchain.tools.base import BaseTool

from AFAAS.interfaces.adapters import CompletionModelFunction
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.interfaces.adapters.embeddings.wrapper import DocumentType
from AFAAS.lib.sdk.logger import AFAASLogger

# from AFAAS.interfaces.agent.main import BaseAgent
LOG = AFAASLogger(name=__name__)


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
        categories: list[str],
        exec_function: Callable[..., ToolOutput],
        parameters: list[ToolParameter],
        success_check_callback: Callable[..., Any],
        enabled: Literal[True] | Callable[[Any], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
        available: Literal[True] | Callable[[BaseAgent], bool] = True,
        tech_description: Optional[str] = None,
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
        self.tech_description = tech_description or description
        self.categories = categories

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.method)

    def __call__(self, *args, agent: BaseAgent, **kwargs) -> Any:
        return self.method(*args, **kwargs, agent=agent)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {'string' if param.spec.type is None and param.spec.required else 'Optional[string]' if param.spec.type is None else param.spec.type.value if param.spec.required else f'Optional[{param.spec.type.value}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description.rstrip('.')}. Params: ({', '.join(params)})"

    @classmethod
    def generate_from_langchain_tool(
        cls,
        langchain_tool: BaseTool,
        arg_converter: Optional[Callable] = None,
        categories: list[str] = ["undefined"],
        success_check_callback=None,
    ) -> "Tool":
        success_check_callback = (
            success_check_callback or cls.default_success_check_callback
        )

        async def wrapper(*args, **kwargs):
            # Remove 'agent' from kwargs if present
            agent = kwargs["agent"]
            # Convert arguments
            tool_input = arg_converter(kwargs, agent) if arg_converter else kwargs

            LOG.debug(f"Running LangChain tool {langchain_tool.name} with arguments {kwargs}")

            # Check if the tool's run method is asynchronous and call accordingly
            if asyncio.iscoroutinefunction(langchain_tool.__call__):
                return await langchain_tool.arun(tool_input)
            else:
                return langchain_tool.run(tool_input)

        typed_parameters = [
            ToolParameter(
                name=name,
                spec=JSONSchema(
                    type=schema.get("type"),
                    description=schema.get("description", schema.get("title")),
                    required=bool(
                        langchain_tool.args_schema.__fields__[name].required
                    )  # gives True if `field.required == pydantic.Undefined``
                    if langchain_tool.args_schema
                    else True,
                ),
            )
            for name, schema in langchain_tool.args.items()
        ]

        _tool_instance = Tool(
            categories=categories,
            name=langchain_tool.name,
            description=langchain_tool.description,
            tech_description=langchain_tool.description,  # Added this line
            exec_function=wrapper,
            parameters=typed_parameters,
            enabled=True,
            disabled_reason=None,
            aliases=[],
            available=True,
            hide=False,
            # Add other optional parameters as needed, like disabled_reason, aliases, etc.
            success_check_callback=success_check_callback,  # Added this line
        )

        # Avoid circular import
        from AFAAS.core.tools.tool_decorator import TOOL_WRAPPER_MARKER

        # Set attributes on the command so that our import module scanner will recognize it
        setattr(_tool_instance, TOOL_WRAPPER_MARKER, True)
        setattr(_tool_instance, "tool", _tool_instance)

        return _tool_instance

    async def default_success_check_callback(
        self, task: AbstractTask, tool_output: Any
    ):
        LOG.trace(f"Tool.default_success_check_callback() called for {self}")
        LOG.debug(f"Task = {task}")
        LOG.debug(f"Tool output = {tool_output}")

        agent: BaseAgent = task.agent

        strategy_result = await agent.execute_strategy(
            strategy_name="afaas_task_default_summary",
            task=task,
            tool_output=tool_output,
            tool=self,
            documents=[],
        )

        task.task_text_output = strategy_result.parsed_result[0]["command_args"][
            "text_output"
        ]
        task.task_text_output_as_uml = strategy_result.parsed_result[0][
            "command_args"
        ].get("text_output_as_uml", "")

        # vector = await agent.vectorstores["tasks"].aadd_texts(
        #     texts=[task.task_text_output],
        #     metadatas=[{"task_id": task.task_id, 
        #                 "plan_id": task.plan_id ,
        #                 "agent_id": task.agent.agent_id ,
        #                 "type" : DocumentType.TASK.value
        #                 }],
        # )
        from langchain_core.documents import Document
        document = Document(
            page_content=task.task_text_output,
            metadata=   {
                        "task_id": task.task_id, 
                        "plan_id": task.plan_id ,
                        "agent_id": task.agent.agent_id ,
                        }
        )
        vector = await agent.vectorstores.add_document(
                                                       document_type = DocumentType.TASK,  
                                                       document = document , 
                                                       document_id =  task.task_id
                                                       ) 


        LOG.trace(f"Task output embedding added to vector store : {repr(vector)}")

        return task.task_text_output

    def dump(self) -> CompletionModelFunction:
        param_dict = {parameter.name: parameter.spec for parameter in self.parameters}

        return CompletionModelFunction(
            name=self.name,
            description=self.description,
            parameters=param_dict,
        )
