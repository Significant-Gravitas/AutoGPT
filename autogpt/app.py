""" Command and Control """
import json
from typing import Dict

from autogpt.agent.agent import Agent
from autogpt.config import Config
from autogpt.llm import ChatModelResponse


def is_valid_int(value: str) -> bool:
    """Check if the value is a valid integer

    Args:
        value (str): The value to check

    Returns:
        bool: True if the value is a valid integer, False otherwise
    """
    try:
        int(value)
        return True
    except ValueError:
        return False


def extract_command(
    assistant_reply_json: Dict, assistant_reply: ChatModelResponse, config: Config
):
    """Parse the response and return the command name and arguments

    Args:
        assistant_reply_json (dict): The response object from the AI
        assistant_reply (ChatModelResponse): The model response from the AI
        config (Config): The config object

    Returns:
        tuple: The command name and arguments

    Raises:
        json.decoder.JSONDecodeError: If the response is not valid JSON

        Exception: If any other error occurs
    """
    if config.openai_functions:
        if assistant_reply.function_call is None:
            return "Error:", "No 'function_call' in assistant reply"
        assistant_reply_json["command"] = {
            "name": assistant_reply.function_call.name,
            "args": json.loads(assistant_reply.function_call.arguments),
        }
    try:
        if "command" not in assistant_reply_json:
            return "Error:", "Missing 'command' object in JSON"

        if not isinstance(assistant_reply_json, dict):
            return (
                "Error:",
                f"The previous message sent was not a dictionary {assistant_reply_json}",
            )

        command = assistant_reply_json["command"]
        if not isinstance(command, dict):
            return "Error:", "'command' object is not a dictionary"

        if "name" not in command:
            return "Error:", "Missing 'name' field in 'command' object"

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error:", "Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error:", str(e)


def execute_command(
    command_name: str,
    arguments: dict[str, str],
    agent: Agent,
):
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command
        agent (Agent): The agent that is executing the command

    Returns:
        str: The result of the command
    """
    try:
        # Execute a native command with the same name or alias, if it exists
        if command := agent.command_registry.get_command(command_name):
            return command(**arguments, agent=agent)

        # Handle non-native commands (e.g. from plugins)
        for command in agent.ai_config.prompt_generator.commands:
            if (
                command_name == command["label"].lower()
                or command_name == command["name"].lower()
            ):
                return command["function"](**arguments)

        raise RuntimeError(
            f"Cannot execute '{command_name}': unknown command."
            " Do not try to use this command again."
        )
    except Exception as e:
        return f"Error: {str(e)}"
