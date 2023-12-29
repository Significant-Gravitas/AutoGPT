from typing import Any, Callable, Optional

from langchain.tools import BaseTool

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.models.command_parameter import CommandParameter


class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., Any],
        parameters: list[CommandParameter],
        enabled: bool | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
    ):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.aliases = aliases

    def __call__(self, *args, **kwargs) -> Any:
        if hasattr(kwargs, "config") and callable(self.enabled):
            self.enabled = self.enabled(kwargs["config"])
        if not self.enabled:
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.type if param.required else f'Optional[{param.type}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description}, params: ({', '.join(params)})"

    @classmethod
    def generate_from_langchain_tool(
        cls, tool: BaseTool, arg_converter: Optional[Callable] = None
    ) -> "Command":
        def wrapper(*args, **kwargs):
            # a Tool's run function doesn't take an agent as an arg, so just remove that
            agent = kwargs.pop("agent")

            # Allow the command to do whatever arg conversion it needs
            if arg_converter:
                tool_input = arg_converter(kwargs, agent)
            else:
                tool_input = kwargs

            logger.debug(f"Running LangChain tool {tool.name} with arguments {kwargs}")

            return tool.run(tool_input=tool_input)

        command = cls(
            name=tool.name,
            description=tool.description,
            method=wrapper,
            parameters=[
                CommandParameter(
                    name=name,
                    type=schema.get("type"),
                    description=schema.get("description", schema.get("title")),
                    required=bool(
                        tool.args_schema.__fields__[name].required
                    )  # gives True if `field.required == pydantic.Undefined``
                    if tool.args_schema
                    else True,
                )
                for name, schema in tool.args.items()
            ],
        )

        # Avoid circular import
        from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER

        # Set attributes on the command so that our import module scanner will recognize it
        setattr(command, AUTO_GPT_COMMAND_IDENTIFIER, True)
        setattr(command, "command", command)

        return command
