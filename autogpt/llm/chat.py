from __future__ import annotations

import time
from random import shuffle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agent.agent import Agent

from autogpt.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import ChatPrompt, Message
from autogpt.llm.utils import count_message_tokens, create_chat_completion
from autogpt.log_cycle.log_cycle import CURRENT_CONTEXT_FILE_NAME
from autogpt.logs import logger
from autogpt.memory.vector import MemoryItem, VectorMemory

cfg = Config()


def create_chat_message(role: str, content: str) -> Message:
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}


# TODO: Change debug from hardcode to argument
def chat_with_ai(
    agent: Agent,
    system_prompt: str,
    user_input: str,
    token_limit: int,
):
    """
    Interact with the OpenAI API, sending the prompt, user input,
        message history, and permanent memory.

    Args:
        system_prompt (str): The prompt explaining the rules to the AI.
        user_input (str): The input from the user.
        token_limit (int): The maximum number of tokens allowed in the API call.

    Returns:
    str: The AI's response.
    """
    model = cfg.fast_llm_model  # TODO: Change model from hardcode to argument
    # Reserve 1000 tokens for the response
    logger.debug(f"Token limit: {token_limit}")
    send_token_limit = token_limit - 1000

    # if len(agent.history) == 0:
    #     relevant_memory = ""
    # else:
    #     recent_history = agent.history[-5:]
    #     shuffle(recent_history)
    #     relevant_memories = agent.memory.get_relevant(
    #         str(recent_history), 5
    #     )
    #     if relevant_memories:
    #         shuffle(relevant_memories)
    #     relevant_memory = str(relevant_memories)
    # logger.debug(f"Memory Stats: {agent.memory.get_stats()}")
    relevant_memory = []

    message_chain = ChatPrompt.for_model(
        model,
        [
            create_chat_message("system", system_prompt),
            create_chat_message(
                "system", f"The current time and date is {time.strftime('%c')}"
            ),
            # create_chat_message(
            #     "system",
            #     f"This reminds you of these events from your past:\n{relevant_memory}\n\n",
            # ),
        ],
    )

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(agent.history) - 1
    insertion_index = len(message_chain)
    # Count the currently used tokens
    current_tokens_used = message_chain.token_length

    # while current_tokens_used > 2500:
    #     # remove memories until we are under 2500 tokens
    #     relevant_memory = relevant_memory[:-1]
    #     (
    #         next_message_to_add_index,
    #         current_tokens_used,
    #         insertion_index,
    #         current_context,
    #     ) = generate_context(
    #         prompt, relevant_memory, agent.history, model
    #     )

    # Account for user input (appended later)
    user_input_msg = create_chat_message("user", user_input)
    current_tokens_used += count_message_tokens([user_input_msg], model)

    current_tokens_used += 500  # Account for memory (appended later) TODO: The final memory may be less than 500 tokens

    # Add Messages until the token limit is reached or there are no more messages to add.
    while next_message_to_add_index >= 0:
        message_to_add = agent.history[next_message_to_add_index]

        tokens_to_add = count_message_tokens([message_to_add], model)
        if current_tokens_used + tokens_to_add > send_token_limit:
            break

        # Add the most recent message to the start of the chain,
        #  after the system prompts.
        message_chain.insert(insertion_index, agent.history[next_message_to_add_index])
        current_tokens_used += tokens_to_add
        next_message_to_add_index -= 1

    # Update & add summary of trimmed messages
    if len(agent.history) > 0:
        new_summary_message, newly_trimmed_messages = agent.history.trim_messages(
            current_message_chain=list(message_chain),
        )
        message_chain.insert(insertion_index, new_summary_message)

    api_manager = ApiManager()
    # inform the AI about its remaining budget (if it has one)
    if api_manager.get_total_budget() > 0.0:
        remaining_budget = api_manager.get_total_budget() - api_manager.get_total_cost()
        if remaining_budget < 0:
            remaining_budget = 0
        budget_message = f"Your remaining API budget is ${remaining_budget:.3f}" + (
            " BUDGET EXCEEDED! SHUT DOWN!\n\n"
            if remaining_budget == 0
            else " Budget very nearly exceeded! Shut down gracefully!\n\n"
            if remaining_budget < 0.005
            else " Budget nearly exceeded. Finish up.\n\n"
            if remaining_budget < 0.01
            else "\n\n"
        )
        logger.debug(budget_message)
        message_chain.add("system", budget_message)

    # Append user input, the length of this is accounted for above
    message_chain.append(user_input_msg)

    plugin_count = len(cfg.plugins)
    for i, plugin in enumerate(cfg.plugins):
        if not plugin.can_handle_on_planning():
            continue
        plugin_response = plugin.on_planning(
            agent.config.prompt_generator, list(message_chain)
        )
        if not plugin_response or plugin_response == "":
            continue
        tokens_to_add = count_message_tokens(
            [create_chat_message("system", plugin_response)], model
        )
        if current_tokens_used + tokens_to_add > send_token_limit:
            logger.debug(f"Plugin response too long, skipping: {plugin_response}")
            logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
            break
        message_chain.add("system", plugin_response)

    # Calculate remaining tokens
    tokens_remaining = token_limit - current_tokens_used
    # assert tokens_remaining >= 0, "Tokens remaining is negative.
    # This should never happen, please submit a bug report at
    #  https://www.github.com/Torantulino/Auto-GPT"

    # Debug print the current context
    logger.debug(f"Token limit: {token_limit}")
    logger.debug(f"Send Token Count: {current_tokens_used}")
    logger.debug(f"Tokens remaining for response: {tokens_remaining}")
    logger.debug("------------ CONTEXT SENT TO AI ---------------")
    for message in message_chain:
        # Skip printing the prompt
        if message["role"] == "system" and message["content"] == system_prompt:
            continue
        logger.debug(f"{message['role'].capitalize()}: {message['content']}")
        logger.debug("")
    logger.debug("----------- END OF CONTEXT ----------------")
    agent.log_cycle_handler.log_cycle(
        agent.config.ai_name,
        agent.created_at,
        agent.cycle_count,
        message_chain,
        CURRENT_CONTEXT_FILE_NAME,
    )

    # TODO: use a model defined elsewhere, so that model can contain
    # temperature and other settings we care about
    assistant_reply = create_chat_completion(
        model=model,
        messages=list(message_chain),
        max_tokens=tokens_remaining,
    )

    # Update full message history
    agent.history.append(user_input_msg)
    agent.history.add("assistant", assistant_reply)

    return assistant_reply
