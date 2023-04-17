""" Command and Control """
import json
from typing import Any, Callable, Dict, List, NoReturn, Optional, Union

from autogpt.agent.agent_manager import AgentManager
from autogpt.commands import *
from autogpt.config import Config
from autogpt.json_fixes.parsing import fix_and_parse_json
from autogpt.memory import get_memory
from autogpt.processing.text import summarize_text
from autogpt.speech import say_text

CFG = Config()
AGENT_MANAGER = AgentManager()
COMMANDS: Dict[str, Callable] = {}


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


def execute_command(command_name: str, arguments: Dict[str, Any]):
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command

    Returns:
        str: The result of the command"""
    command = COMMANDS.get(command_name)

    if command is None:
        return (
            f"Unknown command '{command_name}'. Please refer to the 'COMMANDS'"
            " list for available commands and only respond in the specified JSON"
            " format."
        )

    try:
        return command(**arguments)
    except Exception as e:
        return f"Error: {str(e)}"


def command(name: Optional[str] = None, aliases: list[str] = []):
    """Decorator to register a function as a command

    Args:
        func (function): The function to register as a command

    Returns:
        function: The registered function"""

    def decorator(func):
        if name is None:
            COMMANDS[func.__name__] = func
        else:
            COMMANDS[name] = func

        for alias in aliases:
            COMMANDS[alias] = func

        return func

    return decorator


@command(aliases=["search"])
def google(input: str):
    # Check if the Google API key is set and use the official search method
    # If the API key is not set or has only whitespaces, use the unofficial
    # search method
    key = CFG.google_api_key
    if key and key.strip() and key != "your-google-api-key":
        google_result = google_search.google_official_search(input)
        return google_result
    else:
        google_result = google_search.google_search(input)

    # google_result can be a list or a string depending on the search results
    if isinstance(google_result, list):
        safe_message = [
            google_result_single.encode("utf-8", "ignore")
            for google_result_single in google_result
        ]
    else:
        safe_message = google_result.encode("utf-8", "ignore")

    return safe_message.decode("utf-8")


@command()
def memory_add(string: str) -> str:
    """Add a string to memory

    Args:
        string (str): The string to add to memory

    Returns:
        str: The string that was added to memory"""
    memory = get_memory(CFG)
    return memory.add(string)


@command()
def clone_repository(repository_url: str, clone_path: str) -> str:
    """Clone a repository

    Args:
        url (str): The URL of the repository to clone

    Returns:
        str: The URL of the repository that was cloned"""
    return git_operations.clone_repository(repository_url, clone_path)


@command()
def read_file(file: str) -> str:
    """Read a file

    Args:
        file (str): The file to read

    Returns:
        str: The contents of the file"""
    return file_operations.read_file(file)


@command(aliases=["write_file", "create_file"])
def write_to_file(file: str, text: str) -> str:
    """Write to a file

    Args:
        file (str): The file to write to
        text (str): The text to write to the file

    Returns:
        str: The text that was written to the file"""
    return file_operations.write_to_file(file, text)


@command(aliases=["append_file"])
def append_to_file(file: str, text: str) -> str:
    """Append to a file

    Args:
        file (str): The file to append to
        text (str): The text to append to the file

    Returns:
        str: The text that was appended to the file"""
    return file_operations.append_to_file(file, text)


@command()
def delete_file(file: str) -> str:
    """Delete a file

    Args:
        file (str): The file to delete

    Returns:
        str: The result of the deletion"""
    return file_operations.delete_file(file)


@command()
def search_files(directory: str) -> list[str]:
    """Search for files in a directory

    Args:
        directory (str): The directory to search

    Returns:
        list[str]: The list of files in the directory"""
    return file_operations.search_files(directory)


@command()
def download_file(url: str, file: str) -> str:
    """Download a file from a URL

    Args:
        url (str): The URL to download the file from
        file (str): The file to save the downloaded file to

    Returns:
        str: The result of the download"""
    if not CFG.allow_downloads:
        return "Error: You do not have user authorization to download files locally."
    return file_operations.download_file(url, file)


@command()
def browse_website(url: str, question: str):
    """Browse a website and answer a question

    Args:
        url (str): The URL to browse
        question (str): The question to answer

    Returns:
        str: The answer to the question"""
    return web_selenium.browse_website(url, question)


@command(name="analyze_code")
def do_analyze_code(code: str) -> list[str]:
    """Analyze a piece of code

    Args:
        code (str): The code to analyze

    Returns:
        str: The result of the analysis"""
    return analyze_code.analyze_code(code)


@command(name="improve_code")
def do_improve_code(suggestions: list[str], code: str) -> str:
    """Improve a piece of code

    Args:
        suggestions (list[str]): The suggestions to improve the code
        code (str): The code to improve

    Returns:
        str: The improved code"""
    return improve_code.improve_code(suggestions, code)


@command(name="write_tests")
def do_write_tests(code: str, focus: list[str] = []) -> str:
    """Write tests for a piece of code

    Args:
        code (str): The code to write tests for
        focus (str, optional): The focus of the tests. Defaults to None.

    Returns:
        str: The tests for the code"""
    return write_tests.write_tests(code, focus)


@command()
def execute_python_file(file: str) -> str:
    """Execute a python file

    Args:
        file (str): The file to execute

    Returns:
        str: The result of the execution"""
    return execute_code.execute_python_file(file)


@command()
def execute_shell(command_line: str) -> str:
    """Execute a shell command

    Args:
        command_line (str): The command to execute

    Returns:
        str: The result of the execution"""
    if not CFG.execute_local_commands:
        return (
            "You are not allowed to run local shell commands. To execute"
            " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
            "in your config. Do not attempt to bypass the restriction."
        )
    return execute_code.execute_shell(command_line)


@command()
def execute_shell_popen(command_line: str) -> str:
    """Execute a shell command with Popen

    Args:
        command_line (str): The command to execute

    Returns:
        str: The result of the execution"""
    if not CFG.execute_local_commands:
        return (
            "You are not allowed to run local shell commands. To execute"
            " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
            "in your config. Do not attempt to bypass the restriction."
        )
    return execute_code.execute_shell_popen(command_line)


@command()
def read_audio_from_file(file: str) -> str:
    """Read audio from a file

    Args:
        file (str): The file to read the audio from

    Returns:
        str: The audio from the file"""
    return audio_text.read_audio_from_file(file)


@command()
def generate_image(prompt: str) -> str:
    """Generate an image

    Args:
        prompt (str): The prompt to generate the image from

    Returns:
        str: The generated image"""
    return image_gen.generate_image(prompt)


@command()
def send_tweet(text: str) -> None:
    """Send a tweet

    Args:
        text (str): The text to send as a tweet

    Returns:
        str: The result of the tweet"""
    return twitter.send_tweet(text)


@command()
def do_nothing() -> str:
    return "No action performed."


@command()
def get_text_summary(url: str, question: str) -> str:
    """Return the results of a Google search

    Args:
        url (str): The url to scrape
        question (str): The question to summarize the text for

    Returns:
        str: The summary of the text
    """
    text = web_requests.scrape_text(url)
    summary = summarize_text(url, text, question)
    return f""" "Result" : {summary}"""


@command()
def get_hyperlinks(url: str) -> Union[str, List[str]]:
    """Return the results of a Google search

    Args:
        url (str): The url to scrape

    Returns:
        str or list: The hyperlinks on the page
    """
    return web_requests.scrape_links(url)


@command(aliases=["task_complete"])
def shutdown() -> NoReturn:
    """Shut down the program"""
    print("Shutting down...")
    quit()


@command()
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


@command()
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


@command()
def list_agents():
    """List all agents

    Returns:
        str: A list of all agents
    """
    return "List of agents:\n" + "\n".join(
        [str(x[0]) + ": " + x[1] for x in AGENT_MANAGER.list_agents()]
    )


@command()
def delete_agent(key: str) -> str:
    """Delete an agent with a given key

    Args:
        key (str): The key of the agent to delete

    Returns:
        str: A message indicating whether the agent was deleted or not
    """
    result = AGENT_MANAGER.delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."
