from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from AFAAS.interfaces.tools.tool_output import ToolOutput
from AFAAS.interfaces.tools.tool_parameters import ToolParameter

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from langchain.tools.base import BaseTool

from AFAAS.interfaces.task.task import AbstractTask
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

    @classmethod
    def generate_from_langchain_tool(
        cls, tool: BaseTool, arg_converter: Optional[Callable] = None
    ) -> "Tool":
        def wrapper(*args, **kwargs):
            # a Tool's run function doesn't take an agent as an arg, so just remove that
            agent = kwargs.pop("agent")

            # Allow the command to do whatever arg conversion it needs
            if arg_converter:
                tool_input = arg_converter(kwargs, agent)
            else:
                tool_input = kwargs

            LOG.debug(f"Running LangChain tool {tool.name} with arguments {kwargs}")

            return tool.run(tool_input=tool_input)

        typed_parameters = [
            ToolParameter(
                name=name,
                spec=schema,
            )
            for name, schema in tool.args.items()
        ]

        # typed_parameters = [
        #         ToolParameter(
        #             name=name,
        #             type=schema.get("type"),
        #             description=schema.get("description", schema.get("title")),
        #             required=bool(
        #                 tool.args_schema.__fields__[name].required
        #             )  # gives True if `field.required == pydantic.Undefined``
        #             if tool.args_schema
        #             else True,
        #         )
        #         for name, schema in tool.args.items()
        #     ]

        command = Tool(
            name=tool.name,
            description=tool.description,
            tech_description=tool.description,  # Added this line
            exec_function=wrapper,
            parameters=typed_parameters,
            enabled=True,
            disabled_reason=None,
            aliases=[],
            available=True,
            hide=False,
            # Add other optional parameters as needed, like disabled_reason, aliases, etc.
            success_check_callback=Tool.default_success_check_callback,  # Added this line
        )

        # Avoid circular import
        from AFAAS.core.tools.tool_decorator import AFAAS_TOOL_IDENTIFIER

        # Set attributes on the command so that our import module scanner will recognize it
        setattr(command, AFAAS_TOOL_IDENTIFIER, True)
        setattr(command, "tool", command)

        return command

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

        # task_ouput_embedding = await agent.embedding_model.aembed_query(text = task.task_text_output)
        # vector = await agent.vectorstore.aadd_texts(
        #     task_ouput_embedding, metadatas= [{'task_id' : task.task_id , 'plan_id' : task.plan_id}]
        #     )
        vector = await agent.vectorstores["tasks"].aadd_texts(
            texts=[task.task_text_output],
            metadatas=[{"task_id": task.task_id, "plan_id": task.plan_id}],
        )

        LOG.trace(f"Task output embedding added to vector store : {repr(vector)}")

        return task.task_text_output

        # return summary
