import re


def is_action_auto_gpt(log):
    """AutoGPTs actions are defined by the presence of the "command" key."""
    return bool(re.search(r'"command"\s*:', log))


"""Other notes
Performing actions
- web_search
- write_to_file
- browse_website
Internal actions
- goals_accomplished
"""


def is_openai_function(log):
    """OpenAI API function calls are determined by the presence of the "function_call" key."""
    return bool(re.search(r'"function_call"\s*:', log))


"""KEYWORDS FOUND SO FAR
WRITE
- write
- start
- create
MODIFY
- modify
- mutate
- delete
SEARCH
- search
- find
- get
- browse
READ
- read
GENERAL, no specificity
- command
- call
- function
"""
