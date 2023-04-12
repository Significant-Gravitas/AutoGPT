import datetime
import json
import subprocess

import agent_manager as agents
import ai_functions as ai
import browse
import requests
from config import Config
from duckduckgo_search import ddg
from execute_code import execute_python_file
from file_operations import (
    append_to_file,
    delete_file,
    read_file,
    search_files,
    write_to_file,
)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from image_gen import generate_image
from json_parser import fix_and_parse_json
from memory import get_memory

cfg = Config()


def is_valid_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def get_command(response):
    """Parse the response and return the command name and arguments"""
    try:
        response_json = fix_and_parse_json(response)

        if "command" not in response_json:
            return "Error:", "Missing 'command' object in JSON"

        command = response_json["command"]

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


COMMAND_FUNCTIONS = {
    "google": lambda args: ddg_search(args["query"]),
    "memory_add": lambda args: get_memory(cfg).add(args["string"]),
    "start_agent": lambda args: start_agent(args["name"], args["task"], args["prompt"]),
    "message_agent": lambda args: message_agent(args["key"], args["message"]),
    "list_agents": lambda _: list_agents(),
    "delete_agent": lambda args: delete_agent(args["key"]),
    "get_text_summary": lambda args: get_text_summary(args["url"], args["question"]),
    "get_hyperlinks": lambda args: get_hyperlinks(args["url"]),
    "read_file": lambda args: read_file(args["file"]),
    "write_to_file": lambda args: write_to_file(args["file"], args["text"]),
    "append_to_file": lambda args: append_to_file(args["file"], args["text"]),
    "delete_file": lambda args: delete_file(args["file"]),
    "search_files": lambda args: search_files(args["directory"]),
    "browse_website": lambda args: browse_website(args["url"], args["question"]),
    "evaluate_code": lambda args: ai.evaluate_code(args["code"]),
    "improve_code": lambda args: ai.improve_code(args["suggestions"], args["code"]),
    "write_tests": lambda args: ai.write_tests(args["code"], args.get("focus")),
    "generate_image": lambda args: generate_image(args["prompt"]),
    "execute_python_file": lambda args: execute_python_file(args["file"]),
    "execute_local_command": lambda args: execute_local_command(args["command"]),
    "do_nothing": lambda _: "No action performed.",
    "task_complete": lambda _: shutdown(),
    "api_call": lambda args: api_call(
        args["url"], args["method"], args["headers"], args["body"]
    ),
}


def execute_command(command_name, arguments):
    """Execute the command and return the result."""
    command_function = COMMAND_FUNCTIONS.get(command_name)

    if not command_function:
        return f"Unknown command '{command_name}'. Please refer to the 'COMMANDS' list for available commands and only respond in the specified JSON format."

    try:
        return command_function(arguments)
    except Exception as e:
        return f"Error: {str(e)}"


def get_datetime():
    """Return the current date and time"""
    return "Current date and time: " + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def ddg_search(query, num_results=8):
    """Return the results of a google search"""
    search_results = list(ddg(query, max_results=num_results))
    return json.dumps(search_results, ensure_ascii=False, indent=4)


def browse_website(url, question):
    """Browse a website and return the summary and links"""
    summary = get_text_summary(url, question)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    return f"""Website Content Summary: {summary}\n\nLinks: {links}"""


def get_text_summary(url, question):
    """Return the results of a google search"""
    text = browse.scrape_text(url)
    summary = browse.summarize_text(text, question)
    return f""" "Result" : {summary}"""


def get_hyperlinks(url):
    """Return the results of a google search"""
    return browse.scrape_links(url)


def commit_memory(string):
    """Commit a string to memory"""
    _text = f"""Committing memory with string "{string}" """
    mem.permanent_memory.append(string)
    return _text


def delete_memory(key):
    """Delete a memory with a given key"""
    if key >= 0 and key < len(mem.permanent_memory):
        _text = f"Deleting memory with key {str(key)}"
        del mem.permanent_memory[key]
        print(_text)
        return _text
    else:
        print("Invalid key, cannot delete memory.")
        return None


def overwrite_memory(key, string):
    """Overwrite a memory with a given key and string"""
    # Check if the key is a valid integer
    if is_valid_int(key):
        key_int = int(key)
        # Check if the integer key is within the range of the permanent_memory list
        if 0 <= key_int < len(mem.permanent_memory):
            _text = f"Overwriting memory with key {str(key)} and string {string}"
            return _extracted_from_overwrite_memory_10(string, key_int, _text)
        else:
            print(f"Invalid key '{key}', out of range.")
            return None
    elif isinstance(key, str):
        _text = f"Overwriting memory with key {key} and string {string}"
        return _extracted_from_overwrite_memory_10(string, key, _text)
    else:
        print(f"Invalid key '{key}', must be an integer or a string.")
        return None


# TODO Rename this here and in `overwrite_memory`
def _extracted_from_overwrite_memory_10(string, arg1, _text):
    # Overwrite the memory slot with the given integer key and string
    mem.permanent_memory[arg1] = string
    print(_text)
    return _text


def shutdown():
    """Shut down the program"""
    print("Shutting down...")
    quit()


def start_agent(name, task, prompt, model=cfg.fast_llm_model):
    """Start an agent with a given name, task, and prompt"""
    global cfg

    # Remove underscores from name
    voice_name = name.replace("_", " ")

    first_message = f"""You are {name}.  Respond with: "Acknowledged"."""
    agent_intro = f"{voice_name} here, Reporting for duty!"

    key, ack = agents.create_agent(task, first_message, model)

    # Assign task (prompt), get response
    agent_response = message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


def message_agent(key, message):
    """Message an agent with a given key and message"""
    global cfg

    # Check if the key is a valid integer
    if is_valid_int(key):
        agent_response = agents.message_agent(int(key), message)
    # Check if the key is a valid string
    elif isinstance(key, str):
        agent_response = agents.message_agent(key, message)
    else:
        return "Invalid key, must be an integer or a string."

    return agent_response


def list_agents():
    """List all agents"""
    return agents.list_agents()


def delete_agent(key):
    """Delete an agent with a given key"""
    result = agents.delete_agent(key)
    return f"Agent {key} deleted." if result else f"Agent {key} does not exist."


def execute_local_command(command):
    """Execute a local command"""
    # Check if the command is a valid string
    if isinstance(command, str):
        # Execute the command and return the output
        return subprocess.check_output(command, shell=True).decode("utf-8")
    else:
        return "Invalid command, must be a string."


def api_call(url, method="GET", headers=None, body=None):
    """Make an API call with a given URL, method, and body"""
    return requests.request(method=method, url=url, headers=headers, data=body).text
