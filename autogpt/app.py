""" Command and Control """
import json
from typing import Dict, List, NoReturn, Union
from autogpt import memory

from autogpt.agent.agent_manager import AgentManager
from autogpt.commands.analyze_code import analyze_code
from autogpt.commands.audio_text import read_audio_from_file
from autogpt.commands.execute_code import (
    execute_python_file,
    execute_shell,
    execute_shell_popen,
)
from autogpt.commands.file_operations import (
    append_to_file,
    delete_file,
    download_file,
    read_file,
    search_files,
    write_to_file,
)
from autogpt.commands.web_selenium import browse_website
from autogpt.contexts.templates import TemplateManager
from autogpt.commands.google_search import google_official_search, google_search
from autogpt.commands.image_gen import generate_image
from autogpt.commands.web_playwright import scrape_links, scrape_text
from autogpt.config import Config
from autogpt.contexts.contextualize import ContextManager
from autogpt.json_fixes.parsing import fix_and_parse_json
from autogpt.memory import get_memory
from autogpt.processing.text import summarize_text
from autogpt.speech import say_text
from autogpt.commands.git_operations import clone_repository
from autogpt.commands.twitter import send_tweet
from autogpt.workspace import CONTEXTS_PATH
from autogpt.prompt import all_commands


CFG = Config()
AGENT_MANAGER = AgentManager()
context_template_file = CONTEXTS_PATH / "context_template.md"
context_manager = ContextManager(CONTEXTS_PATH, context_template_file)
template_manager = TemplateManager()


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


def execute_command(command_name: str, arguments):
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command

    Returns:
        str: The result of the command
    """
    try:
        command_name = map_command_synonyms(command_name.lower())
        # todo: unecessary pretty sure
        # if command_name == "list_commands":
        #     formatted_commands = "\n".join([f"\"{command_name}\": {args}" for _, command_name, args in all_commands])
        #     return formatted_commands
        # todo: work out multiple commands in one request
        # if command_name == "command_sequence":
        #     try:
        #         commands = arguments["commands"]
        #         if isinstance(commands, str):
        #             commands = json.loads(commands)

        #         results = []
        #         for command in commands:
        #             if isinstance(command, str):
        #                 command = json.loads(command)
        #             command_result = execute_command(command["name"], command["args"])
        #             result_str = f"{command['name']} command result: {command_result}"
        #             results.append(result_str)

        #         return "\n".join(results)
        #     except Exception as e:
        #         return "command_sequence failed. Ensure you provide the commands in proper json with a name and the correct args"


        if command_name == "google":
            # Check if the Google API key is set and use the official search method
            # If the API key is not set or has only whitespaces, use the unofficial
            # search method
            key = CFG.google_api_key
            if key and key.strip() and key != "your-google-api-key":
                google_result = google_official_search(arguments["input"])
                return google_result
            else:
                google_result = google_search(arguments["input"])

            # google_result can be a list or a string depending on the search results
            if isinstance(google_result, list):
                safe_message = [
                    google_result_single.encode("utf-8", "ignore")
                    for google_result_single in google_result
                ]
            else:
                safe_message = google_result.encode("utf-8", "ignore")

            return str(safe_message)
        elif command_name == "browse_website":
            return browse_website(arguments["url"], arguments["question"])
        
        # AGENTS

        elif command_name == "start_agent":
            return start_agent(
                arguments["name"], arguments["task"], arguments["prompt"]
            )
        elif command_name == "message_agent":
            return message_agent(arguments["key"], arguments["message"])
        elif command_name == "list_agents":
            return list_agents()
        elif command_name == "delete_agent":
            return delete_agent(arguments["key"])


        # Templates
        elif command_name == "list_templates": # LIST
            template_manager = TemplateManager()
            return template_manager.list_templates()
        elif command_name == "read_template": # READ
            template_name = arguments["name"]
            template_manager = TemplateManager()
            return template_manager.read_template(template_name)
        elif command_name == "create_template": # CREATE
            template_name = arguments["name"]
            template_data = arguments["data"]
            template_manager = TemplateManager()
            return template_manager.create_template(template_name, template_data)
        
        # CONTEXTS

        elif command_name == "list_contexts": # LIST
            return context_manager.list_contexts()
        elif command_name == "get_current_context": # CURRENT
            return context_manager.get_current_context()
        elif command_name == "create_context": # CREATE
            context_name = arguments["name"]
            context_data = arguments["data"]
            return context_manager.create_new_context(context_name, context_data)
        elif command_name == "evaluate_context": # EVALUATE
            context_name = arguments["name"]
            context_eval = arguments["data"]
            return context_manager.evaluate_context_success(context_name, context_eval)
        elif command_name == "close_context": # CLOSE
            context_name = arguments["name"]
            context_close_summary = arguments["data"]
            return context_manager.close_context(context_name, context_close_summary)
        elif command_name == "switch_context": # SWITCH
            context_name = arguments["name"]
            return context_manager.switch_context(context_name)

        # Todo: Implement these, change args
        elif command_name == "merge_contexts": # MERGE
            context_name_1 = arguments["context_name_1"]
            context_name_2 = arguments["context_name_2"]
            merged_context_name = arguments["merged_context_name"]
            merged_context_data = arguments["merged_context_data"]
            context_manager.merge_contexts(context_name_1, context_name_2, merged_context_name, merged_context_data)
        elif command_name == "update_context": # UPDATE
            context_name = arguments["context_name"]
            context_data = arguments["filled_template_markdown_data"]
            return context_manager.update_context(context_name, context_data)
        elif command_name == "get_context": # GET
            context_name = arguments["context_name"]
            return context_manager.get_context(context_name)

        
        elif command_name == "get_text_summary":
            return get_text_summary(arguments["url"], arguments["question"])
        elif command_name == "get_hyperlinks":
            return get_hyperlinks(arguments["url"])
        elif command_name == "clone_repository":
            return clone_repository(
                arguments["repository_url"], arguments["clone_path"]
            )
        elif command_name == "read_file":
            return read_file(arguments["file"])
        elif command_name == "write_to_file":
            return write_to_file(arguments["file"], arguments["text"])
        elif command_name == "append_to_file":
            return append_to_file(arguments["file"], arguments["text"])
        elif command_name == "delete_file":
            return delete_file(arguments["file"])
        elif command_name == "search_files":
            return search_files(arguments["directory"])
        
        elif command_name == "generate_image":
            return generate_image(arguments["prompt"])
        elif command_name == "do_nothing":
            return "No action performed."
        elif command_name == "task_complete":
            shutdown()

        elif command_name == "memory_add":
            return memory.add(arguments["string"])
        else:
            return (
                f"Unknown command '{command_name}'. Please refer to the 'COMMANDS'"
                " list for available commands and only respond in the specified DSL syntax."
                # Both JSON and DSL syntax seem to prompt correctly :shrug:
            )
    except Exception as e:
        return f"Error: {str(e)}"


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



# AGENT COMMANDS

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


def list_agents():
    """List all agents

    Returns:
        str: A list of all agents
    """
    return "List of agents:\n" + "\n".join(
        [str(x[0]) + ": " + x[1] for x in AGENT_MANAGER.list_agents()]
    )


def delete_agent(key: str) -> str:
    """Delete an agent with a given key

    Args:
        key (str): The key of the agent to delete

    Returns:
        str: A message indicating whether the agent was deleted or not
    """
    result = AGENT_MANAGER.delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."
