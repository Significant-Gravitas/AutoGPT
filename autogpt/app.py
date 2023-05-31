""" Command and Control """
import json
from typing import Dict, List, Union

from autogpt.agent.agent_manager import AgentManager
from autogpt.commands.command import CommandRegistry, command
from autogpt.commands.web_requests import scrape_links, scrape_text
from autogpt.config import Config
from autogpt.processing.text import summarize_text
from autogpt.prompts.generator import PromptGenerator
from autogpt.speech import say_text
from autogpt.url_utils.validators import validate_url


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


def get_command(response_json: Dict):
    """Parse the response and return the command name and arguments

    Args:
        response_json (json): The response from the AI

    Returns:
        tuple: The command name and arguments

    Raises:
        json.decoder.JSONDecodeError: If the response is not valid JSON

        Exception: If any other error occurs
    """
    try:
        if "command" not in response_json:
            return "Error:", "Missing 'command' object in JSON"

        if not isinstance(response_json, dict):
            return "Error:", f"'response_json' object is not dictionary {response_json}"

        command = response_json["command"]
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


def map_command_synonyms(command_name: str):
    """Takes the original command name given by the AI, and checks if the
    string matches a list of common/known hallucinations
    """
    synonyms = [
        ("write_file", "write_to_file"),
        ("create_file", "write_to_file"),
        ("search", "google"),
    ]
    for seen_command, actual_command_name in synonyms:
        if command_name == seen_command:
            return actual_command_name
    return command_name


def execute_command(
    command_registry: CommandRegistry,
    command_name: str,
    arguments,
    prompt: PromptGenerator,
    config: Config,
):
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command

    Returns:
        str: The result of the command
    """
    try:
        cmd = command_registry.commands.get(command_name)

        # If the command is found, call it with the provided arguments
        if cmd:
            return cmd(**arguments, config=config)

        # TODO: Remove commands below after they are moved to the command registry.
        command_name = map_command_synonyms(command_name.lower())

        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again
        for command in prompt.commands:
            if (
                command_name == command["label"].lower()
                or command_name == command["name"].lower()
            ):
                return command["function"](**arguments)
        return (
            f"Unknown command '{command_name}'. Please refer to the 'COMMANDS'"
            " list for available commands and only respond in the specified JSON"
            " format."
        )
    except Exception as e:
        return f"Error: {str(e)}"


@command(
    "get_text_summary", "Get text summary", '"url": "<url>", "question": "<question>"'
)
@validate_url
def get_text_summary(url: str, question: str, config: Config) -> str:
    """Get the text summary of a webpage

    Args:
        url (str): The url to scrape
        question (str): The question to summarize the text for

    Returns:
        str: The summary of the text
    """
    text = scrape_text(url)
    summary, _ = summarize_text(text, question=question)

    return f""" "Result" : {summary}"""


@command("get_hyperlinks", "Get hyperlinks", '"url": "<url>"')
@validate_url
def get_hyperlinks(url: str, config: Config) -> Union[str, List[str]]:
    """Get all hyperlinks on a webpage

    Args:
        url (str): The url to scrape

    Returns:
        str or list: The hyperlinks on the page
    """
    return scrape_links(url, config)


@command(
    "start_agent",
    "Start GPT Agent",
    '"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"',
)
def start_agent(name: str, task: str, prompt: str, config: Config, model=None) -> str:
    """Start an agent with a given name, task, and prompt

    Args:
        name (str): The name of the agent
        task (str): The task of the agent
        prompt (str): The prompt for the agent
        model (str): The model to use for the agent

    Returns:
        str: The response of the agent
    """
    agent_manager = AgentManager()

    # Remove underscores from name
    voice_name = name.replace("_", " ")

    first_message = f"""You are {name}.  Respond with: "Acknowledged"."""
    agent_intro = f"{voice_name} here, Reporting for duty!"

    # Create agent
    if config.speak_mode:
        say_text(agent_intro, 1)
    key, ack = agent_manager.create_agent(task, first_message, model)

    if config.speak_mode:
        say_text(f"Hello {voice_name}. Your task is as follows. {task}.")

    # Assign task (prompt), get response
    agent_response = agent_manager.message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


@command("message_agent", "Message GPT Agent", '"key": "<key>", "message": "<message>"')
def message_agent(key: str, message: str, config: Config) -> str:
    """Message an agent with a given key and message"""
    # Check if the key is a valid integer
    if is_valid_int(key):
        agent_response = AgentManager().message_agent(int(key), message)
    else:
        return "Invalid key, must be an integer."

    # Speak response
    if config.speak_mode:
        say_text(agent_response, 1)
    return agent_response


@command("list_agents", "List GPT Agents", "() -> str")
def list_agents(config: Config) -> str:
    """List all agents

    Returns:
        str: A list of all agents
    """
    return "List of agents:\n" + "\n".join(
        [str(x[0]) + ": " + x[1] for x in AgentManager().list_agents()]
    )


@command("delete_agent", "Delete GPT Agent", '"key": "<key>"')
def delete_agent(key: str, config: Config) -> str:
    """Delete an agent with a given key

    Args:
        key (str): The key of the agent to delete

    Returns:
        str: A message indicating whether the agent was deleted or not
    """
    result = AgentManager().delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."
