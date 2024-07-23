"""Commands to generate images based on text input"""

import inspect
import logging
from typing import Any, Callable, Iterable, Iterator

from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema

MAX_RESULT_LENGTH = 1000
logger = logging.getLogger(__name__)


class CodeFlowExecutionComponent(CommandProvider):
    """A component that provides commands to execute code flow."""

    def __init__(self, get_available_commands: Callable[[], Iterable[Command]]):
        self.get_available_commands = get_available_commands

    def get_commands(self) -> Iterator[Command]:
        yield self.execute_code_flow

    @command(
        parameters={
            "python_code": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The Python code to execute",
                required=True,
            ),
            "plan_text": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The plan to written in a natural language",
                required=False,
            ),
        },
    )
    async def execute_code_flow(self, python_code: str, plan_text: str) -> str:
        """Execute the code flow.

        Args:
            python_code: The Python code to execute
            plan_text: The plan implemented by the given Python code

        Returns:
            str: The result of the code execution
        """
        locals: dict[str, Any] = {}
        locals.update(self._get_available_functions())
        code = f"{python_code}" "\n\n" "exec_output = main()"
        logger.debug(f"Code-Flow Execution code:\n```py\n{code}\n```")
        exec(code, locals)
        result = await locals["exec_output"]
        logger.debug(f"Code-Flow Execution result:\n{result}")
        if inspect.isawaitable(result):
            result = await result

        # limit the result to limit the characters
        if len(result) > MAX_RESULT_LENGTH:
            result = result[:MAX_RESULT_LENGTH] + "...[Truncated, Content is too long]"
        return f"Execution Plan:\n{plan_text}\n\nExecution Output:\n{result}"

    def _get_available_functions(self) -> dict[str, Callable]:
        return {
            name: command
            for command in self.get_available_commands()
            for name in command.names
            if name != self.execute_code_flow.name
        }
