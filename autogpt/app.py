""" Command and Control """
import json
from typing import Dict, List, NoReturn, Union

from autogpt.agent.agent_manager import AgentManager
from autogpt.commands.command import CommandRegistry, command
from autogpt.commands.web_requests import scrape_links, scrape_text
from autogpt.config import Config
from autogpt.memory import get_memory
from autogpt.processing.text import summarize_text
from autogpt.prompts.generator import PromptGenerator
from autogpt.speech import say_text

CFG = Config()
AGENT_MANAGER = AgentManager()


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
            return cmd(**arguments)

        # TODO: Remove commands below after they are moved to the command registry.
        command_name = map_command_synonyms(command_name.lower())

        if command_name == "memory_add":
            return get_memory(CFG).add(arguments["string"])

        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again
        elif command_name == "do_nothing":
            return "No action performed."
        elif command_name == "task_complete":
            shutdown()
        else:
            for command in prompt.commands:
                if command_name == command["label"] or command_name == command["name"]:
                    return command["function"](*arguments.values())
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
def get_text_summary(url: str, question: str) -> str:
    """Return the results of a Google search

    Args:
        url (str): The url to scrape
        question (str): The question to summarize the text for

    Returns:
        str: The summary of the text
    """
    text = scrape_text(url)
    summary = summarize_text(url, text, question)
    return f""" "Result" : {summary}"""


@command("get_hyperlinks", "Get text summary", '"url": "<url>"')
def get_hyperlinks(url: str) -> Union[str, List[str]]:
    """Return the results of a Google search

    Args:
        url (str): The url to scrape

    Returns:
        str or list: The hyperlinks on the page
    """
    return scrape_links(url)


def shutdown() -> NoReturn:
    """Shut down the program"""
    print("Shutting down...")
    quit()


@command(
    "start_agent",
    "Start GPT Agent",
    '"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"',
)
def start_agent(name: str, task: str, prompt: str, model=CFG.fast_llm_model) -> str:
    """Start an agent with a given name, task, and prompt

    Args:
        name (str): The name of the agent
        task (str): The task of the agent
        prompt (str): The prompt for the agent
        model (str): The model to use for the agent

    Returns:
        str: The response of the agent
    """
    # Remove underscores from name
    voice_name = name.replace("_", " ")

    first_message = f"""You are {name}.  Respond with: "Acknowledged"."""
    agent_intro = f"{voice_name} here, Reporting for duty!"

    # Create agent
    if CFG.speak_mode:
        say_text(agent_intro, 1)
    key, ack = AGENT_MANAGER.create_agent(task, first_message, model)

    if CFG.speak_mode:
        say_text(f"Hello {voice_name}. Your task is as follows. {task}.")

    # Assign task (prompt), get response
    agent_response = AGENT_MANAGER.message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


@command("message_agent", "Message GPT Agent", '"key": "<key>", "message": "<message>"')
def message_agent(key: str, message: str) -> str:
    """Message an agent with a given key and message"""
    # Check if the key is a valid integer
    if is_valid_int(key):
        agent_response = AGENT_MANAGER.message_agent(int(key), message)
    else:
        return "Invalid key, must be an integer."

    # Speak response
    if CFG.speak_mode:
        say_text(agent_response, 1)
    return agent_response


@command("list_agents", "List GPT Agents", "")
def list_agents() -> str:
    """List all agents

    Returns:
        str: A list of all agents
    """
    return "List of agents:\n" + "\n".join(
        [str(x[0]) + ": " + x[1] for x in AGENT_MANAGER.list_agents()]
    )


@command("delete_agent", "Delete GPT Agent", '"key": "<key>"')
def delete_agent(key: str) -> str:
    """Delete an agent with a given key

    Args:
        key (str): The key of the agent to delete

    Returns:
        str: A message indicating whether the agent was deleted or not
    """
    result = AGENT_MANAGER.delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."
