import datetime
import json

from duckduckgo_search import ddg

import agent_manager as agents
import ai_functions as ai
import browse
import speak
from config import Config
from execute_code import execute_python_file
from file_operations import read_file, write_to_file, append_to_file, delete_file, search_files
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


ERROR_MSG_BEGIN = "Error:"


def get_command(response: str) -> (str, dict):
    try:
        response_json = fix_and_parse_json(response)

        if "command" not in response_json:
            return ERROR_MSG_BEGIN, "Missing 'command' object in JSON"

        command = response_json["command"]

        if "name" not in command:
            return ERROR_MSG_BEGIN, "Missing 'name' field in 'command' object"

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        if not arguments:
            arguments = {}

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return ERROR_MSG_BEGIN, "Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return ERROR_MSG_BEGIN, str(e)


def execute_command(command_name: str, arguments) -> str:
    """
    Execute the command specified by the user.
    Returns a string containing the result of the command.
    If an error occurs, returns a string containing the error message.
    """
    memory = get_memory(cfg)

    def execute_google():
        if cfg.google_api_key and (cfg.google_api_key.strip() if cfg.google_api_key else None):
            return google_official_search(arguments["input"])
        else:
            return google_search(arguments["input"])

    command_mapping = {
        "google": execute_google,
        "memory_add": lambda: memory.add(arguments["string"]),
        "start_agent": lambda: start_agent(arguments["name"], arguments["task"], arguments["prompt"]),
        "message_agent": lambda: message_agent(arguments["key"], arguments["message"]),
        "list_agents": list_agents,
        "delete_agent": lambda: delete_agent(arguments["key"]),
        "get_text_summary": lambda: get_text_summary(arguments["url"], arguments["question"]),
        "get_hyperlinks": lambda: get_hyperlinks(arguments["url"]),
        "read_file": lambda: read_file(arguments["file"]),
        "write_to_file": lambda: write_to_file(arguments["file"], arguments["text"]),
        "append_to_file": lambda: append_to_file(arguments["file"], arguments["text"]),
        "delete_file": lambda: delete_file(arguments["file"]),
        "search_files": lambda: search_files(arguments["directory"]),
        "browse_website": lambda: browse_website(arguments["url"], arguments["question"]),
        "evaluate_code": lambda: ai.evaluate_code(arguments["code"]),
        "improve_code": lambda: ai.improve_code(arguments["suggestions"], arguments["code"]),
        "write_tests": lambda: ai.write_tests(arguments["code"], arguments.get("focus")),
        "execute_python_file": lambda: execute_python_file(arguments["file"]),
        "generate_image": lambda: generate_image(arguments["prompt"]),
        "task_complete": shutdown,
    }

    try:
        if command_name in command_mapping:
            return command_mapping[command_name]()
        else:
            return f"Unknown command '{command_name}'. Please refer to the 'COMMANDS' list for available commands " \
                   f"and only respond in the specified JSON format."
    except Exception as e:
        return ERROR_MSG_BEGIN + str(e)


def get_datetime():
    return "Current date and time: " + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def google_search(query, num_results=8):
    search_results = []
    for j in ddg(query, max_results=num_results):
        search_results.append(j)

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def google_official_search(query, num_results=8):
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import json

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = cfg.google_api_key
        custom_search_engine_id = cfg.custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = service.cse().list(q=query, cx=custom_search_engine_id, num=num_results).execute()

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get("code") == 403 and "invalid API key" in \
                error_details.get("error", {}).get("message", ""):
            return "Error: The provided Google API key is invalid or missing."
        else:
            return f"Error: {e}"

    # Return the list of search result URLs
    return search_results_links


def browse_website(url, question) -> str:
    summary = get_text_summary(url, question)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result


def get_text_summary(url, question) -> str:
    text = browse.scrape_text(url)
    summary = browse.summarize_text(text, question)
    return """ "Result" : """ + summary


def get_hyperlinks(url):
    link_list = browse.scrape_links(url)
    return link_list


def commit_memory(string):
    _text = f"""Committing memory with string "{string}" """
    mem.permanent_memory.append(string)
    return _text


def delete_memory(key):
    if 0 <= key < len(mem.permanent_memory):
        _text = "Deleting memory with key " + str(key)
        del mem.permanent_memory[key]
        print(_text)
        return _text
    else:
        print("Invalid key, cannot delete memory.")
        return None


def overwrite_memory(key, string):
    # Check if the key is a valid integer
    if is_valid_int(key):
        key_int = int(key)
        # Check if the integer key is within the range of the permanent_memory list
        if 0 <= key_int < len(mem.permanent_memory):
            _text = "Overwriting memory with key " + str(key) + " and string " + string
            # Overwrite the memory slot with the given integer key and string
            mem.permanent_memory[key_int] = string
            print(_text)
            return _text
        else:
            print(f"Invalid key '{key}', out of range.")
            return None
    # Check if the key is a valid string
    elif isinstance(key, str):
        _text = "Overwriting memory with key " + key + " and string " + string
        # Overwrite the memory slot with the given string key and string
        mem.permanent_memory[key] = string
        print(_text)
        return _text
    else:
        print(f"Invalid key '{key}', must be an integer or a string.")
        return None


def shutdown():
    print("Shutting down...")
    quit()


def start_agent(name, task, prompt, model=cfg.fast_llm_model):
    global cfg

    # Remove underscores from name
    voice_name = name.replace("_", " ")

    first_message = f"""You are {name}.  Respond with: "Acknowledged"."""
    agent_intro = f"{voice_name} here, Reporting for duty!"

    # Create agent
    if cfg.speak_mode:
        speak.say_text(agent_intro, 1)
    key, ack = agents.create_agent(task, first_message, model)

    if cfg.speak_mode:
        speak.say_text(f"Hello {voice_name}. Your task is as follows. {task}.")

    # Assign task (prompt), get response
    agent_response = message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


def message_agent(key, message):
    global cfg

    # Check if the key is a valid integer
    if is_valid_int(key):
        agent_response = agents.message_agent(int(key), message)
    # Check if the key is a valid string
    elif isinstance(key, str):
        agent_response = agents.message_agent(key, message)
    else:
        return "Invalid key, must be an integer or a string."

    # Speak response
    if cfg.speak_mode:
        speak.say_text(agent_response, 1)
    return agent_response


def list_agents():
    return agents.list_agents()


def delete_agent(key):
    result = agents.delete_agent(key)
    if not result:
        return f"Agent {key} does not exist."
    return f"Agent {key} deleted."
