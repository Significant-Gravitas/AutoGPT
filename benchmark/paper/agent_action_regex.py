import re
import json


def is_action_auto_gpt(log):
    """AutoGPTs actions are defined by the presence of the "command" key.

    World state actions
    - web_search
    - write_to_file
    - browse_website
    - execute_python_file
    - list_files
    Internal actions
    - goals_accomplished

    Input
        The "content" key of an LLM response
    """

    # Check for the existence of the "command" key in the log
    command_existence = bool(re.search(r'"command"\s*:', log))

    if command_existence:
        # Convert the JSON-like string to a Python dictionary
        log_dict = json.loads(log)

        # Check if the "command" key exists and has a "name" key
        if "command" in log_dict and "name" in log_dict["command"]:
            command_name = log_dict["command"]["name"]

            # List of command names that signify an action
            action_command_names = [
                "web_search",
                "write_to_file",
                "browse_website",
                "execute_python_file",
                "list_files",
            ]

            # Check if the command name matches any in the list
            return command_name in action_command_names

    return False


def is_openai_function(log):
    """OpenAI API function calls are determined by the presence of the "function_call" key.
    Beebot
    World state actions
    - get_html_content
    - read_file
    - write_file
    - wolfram_alpha_query
    - write_python_code
    - execute_python_file
    - google_search
    - wikipedia
    Internal actions
    - get_more_tools
    - exit
    - rewind_actions

    PolyGPT
    World state actions
    - http.sendRequest
    - http.post
    - http.get
    - filesystem.writeFile
    - ethers.sendRpc
    Internal actions
    - LearnWrap
    - InvokeWrap

    Input
        The entire LLM response
    """
    # Check for the existence of the "function_call" key in the log
    function_call_existence = bool(log.get("function_call", None))

    if function_call_existence:
        # Check if the "function_call" key exists and has a "name" key
        if "name" in log["function_call"]:
            function_name = log["function_call"]["name"]

            # List of function names that signify an action
            action_function_names = [
                "get_html_content",
                "read_file",
                "write_file",
                "wolfram_alpha_query",
                "write_python_code",
                "execute_python_file",
                "google_search",
                "wikipedia",
                "http.sendRequest",
                "http.post",
                "http.get",
                "filesystem.writeFile",
                "ethers.sendRpc",
            ]

            # Check if the function name matches any in the list
            return function_name in action_function_names

    return False


def is_action_miniagi(log):
    """Mini-AGI function calls are determined by the presence of different patterns
    World state actions
    - execute_python
    - web_search
    Internal actions
    - done
    - talk_to_user
    """
    # List of function names that signify an action
    action_function_names = ["execute_python", "web_search"]

    # Check for the <c>...</c> pattern and whether it matches any action function names
    c_pattern_match = False
    c_pattern_search = re.search(r"<c>(.*?)<\/c>", log)
    if c_pattern_search:
        c_pattern_match = c_pattern_search.group(1) in action_function_names

    # Check for the "ACTION:" pattern and whether it matches any action function names
    action_pattern_match = False
    action_pattern_search = re.search(r"ACTION:\s*(\w+)\s*(\(x\d+\))?", log)
    if action_pattern_search:
        action_pattern_match = action_pattern_search.group(1) in action_function_names

    return c_pattern_match or action_pattern_match


def is_action_turbo(log):
    """Turbos actions are defined by the presence of the "cmd" key.
    World state actions
    - search
    - www
    - py
    - aol
    - put
    - pyf
    Internal actions
    - end
    """
    # List of function names that signify an action
    action_function_names = ["search", "www", "py", "aol", "put", "pyf"]

    # Check for the "cmd" key pattern and whether its "name" field matches any action function names
    cmd_pattern_match = False
    cmd_pattern_search = re.search(r'"cmd"\s*:\s*{\s*"name"\s*:\s*"(\w+)"', log)
    if cmd_pattern_search:
        cmd_pattern_match = cmd_pattern_search.group(1) in action_function_names

    return cmd_pattern_match


def is_action_general(log):
    """General actions are defined by the presence of specific keywords such as 'write', 'start', 'create', etc."""
    return bool(
        re.search(
            r"\b(write|start|create|execute|post|modify|mutate|delete|put|search|find|get|browse|query|www|read|list)\b",
            log,
        )
    )


# post, get, put, delete
"""KEYWORDS FOUND SO FAR
WRITE
- write
- start
- create
- execute
- post
MODIFY
- modify
- mutate
- delete
- put
SEARCH
- search
- find
- get
- browse
- query
- www
READ
- read
- list
GENERAL, no specificity
- command
- call
- function
- action
"""


def is_action_agent(log, agent="", test="", response=""):
    """Determines if a log contains an action based on patterns from different agents."""
    is_action = False

    if log is None:
        print("Log is None", agent, test, response)
        return is_action

    log_content = log.get("content", "")

    if agent == "auto-gpt":
        is_action = is_action_auto_gpt(log_content)
    elif agent in ["beebot", "polygpt"]:
        is_action = is_openai_function(log)
    elif agent == "miniagi":
        is_action = is_action_miniagi(log_content)
    elif agent == "turbo":
        is_action = is_action_turbo(log_content)

    return is_action
