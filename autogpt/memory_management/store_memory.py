from autogpt.json_utils.utilities import (
    LLM_DEFAULT_RESPONSE_FORMAT,
    is_string_valid_json,
)
from autogpt.logs import logger


def format_memory(assistant_reply, next_message_content):
    # the next_message_content is a variable to stores either the user_input or the command following the assistant_reply
    result = (
        "None" if next_message_content.startswith("Command") else next_message_content
    )
    user_input = (
        "None"
        if next_message_content.startswith("Human feedback")
        else next_message_content
    )

    return f"Assistant Reply: {assistant_reply}\nResult: {result}\nHuman Feedback:{user_input}"


def save_memory_trimmed_from_context_window(
    full_message_history, next_message_to_add_index, permanent_memory
):
    while next_message_to_add_index >= 0:
        message_content = full_message_history[next_message_to_add_index]["content"]
        if is_string_valid_json(message_content, LLM_DEFAULT_RESPONSE_FORMAT):
            next_message = full_message_history[next_message_to_add_index + 1]
            memory_to_add = format_memory(message_content, next_message["content"])
            logger.debug(f"Storing the following memory: {memory_to_add}")
            permanent_memory.add(memory_to_add)

        next_message_to_add_index -= 1
