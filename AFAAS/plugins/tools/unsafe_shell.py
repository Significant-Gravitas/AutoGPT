from __future__ import annotations

import importlib
import os
import os.path
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

from dotenv import load_dotenv
from langchain.vectorstores import VectorStore
from langchain_community.tools.file_management.file_search import FileSearchTool

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.sdk.errors import DuplicateOperationError
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema

# Load environment variables
load_dotenv()

# Retrieve the allowed commands and convert them into a list, or an empty list if not defined
ALLOWED_COMMANDS = os.getenv("ALLOWED_COMMANDS")

if not SAFE_MODE or (SAFE_MODE and ALLOWED_COMMANDS is not None):
    if ALLOWED_COMMANDS is not None:
        ALLOWED_COMMANDS = ALLOWED_COMMANDS.split(",")
    else:
        ALLOWED_COMMANDS = []

    def load_validator(command_name):
        # Construct the path to the validator module
        module_path = (
            Path(__file__).parent / f"shell_commands_validators/{command_name}.py"
        )

        if not module_path.exists():
            return default_arg_validation  # Use the default validator

        spec = importlib.util.spec_from_file_location(
            f"{command_name}_validator", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.validate

    def default_arg_validation(args):
        return True  # A basic validator that always returns True

    @tool(
        name="execute_shell_command",
        description="Execute a command with arguments in the shell",
        parameters={
            "command": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command to execute",
                required=True,
            ),
            "arguments": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The arguments for the command",
                required=False,
            ),
        },
    )
    def execute_shell_command(
        command: str, arguments: str, task: Task, agent: BaseAgent
    ) -> str:
        """Execute a shell command with arguments

        Args:
            command (str): The command to execute
            arguments (str): The arguments for the command

        Returns:
            str: The output of the command or an error message
        """
        # Check if the command is allowed, or all commands are allowed
        if ALLOWED_COMMANDS and command not in ALLOWED_COMMANDS:
            return f"Error: Command '{command}' is not allowed."

        # Split arguments into a list
        args = arguments.split()

        # Dynamically load the argument validator
        arg_validator = load_validator(command)

        # Validate the arguments for the command
        if not arg_validator(args):
            return f"Error: Invalid arguments provided for command '{command}'."

        command_string = f"{command} {' '.join(args)}"

        try:
            process = Popen(command_string, shell=True, stdout=PIPE, stderr=STDOUT)
            output, error = process.communicate()
            return output.decode()
        except Exception as e:
            return f"Error occurred while executing the command: {e}"
