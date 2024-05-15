"""Commands to generate images based on text input"""

import logging
from typing import Iterator

from autogpt.agents.protocols import CommandProvider
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.llm.providers.openai import function_specs_from_commands
from autogpt.models.command import Command

logger = logging.getLogger(__name__)


class CodeFlowExecutionComponent(CommandProvider):
    """A component that provides commands to execute code flow."""

    def __init__(self):
        self._enabled = True
        self.available_functions = {}

    def set_available_functions(self, functions: list[Command]):
        self.available_functions = {
            name: function for function in functions for name in function.names
        }

    def get_commands(self) -> Iterator[Command]:
        yield self.execute_code_flow

    @command(
        parameters={
            "python_code": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The Python code to execute",
                required=True,
            ),
        },
    )
    def execute_code_flow(self, python_code: str) -> str:
        """Execute the code flow.

        Args:
            python_code (str): The Python code to execute
            callables (dict[str, Callable]): The dictionary of [name, callable] pairs to use in the code

        Returns:
            str: The result of the code execution
        """
        code = f"{python_code}\n\nexec_output = main()"
        exec_output = None  # type hinting purposes
        exec(code, {name: func for name, func in self.available_functions.items()})
        return str(exec_output)
