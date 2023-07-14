from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

from autogpt.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import ChatSequence, Message
from autogpt.llm.providers.openai import (
    count_openai_functions_tokens,
    get_openai_command_specs,
)
from autogpt.llm.utils import count_message_tokens, create_chat_completion
from autogpt.logs import CURRENT_CONTEXT_FILE_NAME, logger


# TODO: Change debug from hardcode to argument
def chat_with_ai(
    config: Config,
    agent: Agent,
    system_prompt: str,
    triggering_prompt: str,
    token_limit: int,
    model: str | None = None,
):
    """
    Interact with the OpenAI API, sending the prompt, user input,
        message history, and permanent memory.

    Args:
        config (Config): The config to use.
        agent (Agent): The agent to use.
        system_prompt (str): The prompt explaining the rules to the AI.
        triggering_prompt (str): The input from the user.
        token_limit (int): The maximum number of tokens allowed in the API call.
        model (str, optional): The model to use. By default, the config.smart_llm will be used.

    Returns:
    str: The AI's response.
    """
    if model is None:
        model = config.smart_llm

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

    message_sequence = ChatSequence.for_model(
        model,
        [
            Message("system", system_prompt),
            Message("system", f"The current time and date is {time.strftime('%c')}"),
            # Message(
            #     "system",
            #     f"This reminds you of these events from your past:\n{relevant_memory}\n\n",
            # ),
        ],
    )

    # Count the currently used tokens
    current_tokens_used = message_sequence.token_length
    insertion_index = len(message_sequence)

    # Account for tokens used by OpenAI functions
    openai_functions = None
    if agent.config.openai_functions:
        openai_functions = get_openai_command_specs(agent.command_registry)
        functions_tlength = count_openai_functions_tokens(openai_functions, model)
        current_tokens_used += functions_tlength
        logger.debug(f"OpenAI Functions take up {functions_tlength} tokens in API call")

    # Account for user input (appended later)
    user_input_msg = Message("user", triggering_prompt)
    current_tokens_used += count_message_tokens(user_input_msg, model)

    current_tokens_used += agent.history.max_summary_tlength  # Reserve space
    current_tokens_used += 500  # Reserve space for the openai functions TODO improve

    # Add historical Messages until the token limit is reached
    #  or there are no more messages to add.
    for cycle in reversed(list(agent.history.per_cycle())):
        messages_to_add = [msg for msg in cycle if msg is not None]
        tokens_to_add = count_message_tokens(messages_to_add, model)
        if current_tokens_used + tokens_to_add > send_token_limit:
            break

        # Add the most recent message to the start of the chain,
        #  after the system prompts.
        message_sequence.insert(insertion_index, *messages_to_add)
        current_tokens_used += tokens_to_add

    # Update & add summary of trimmed messages
    if len(agent.history) > 0:
        new_summary_message, trimmed_messages = agent.history.trim_messages(
            current_message_chain=list(message_sequence), config=agent.config
        )
        tokens_to_add = count_message_tokens(new_summary_message, model)
        message_sequence.insert(insertion_index, new_summary_message)
        current_tokens_used += tokens_to_add - agent.history.max_summary_tlength

        # FIXME: uncomment when memory is back in use
        # memory_store = get_memory(config)
        # for _, ai_msg, result_msg in agent.history.per_cycle(trimmed_messages):
        #     memory_to_add = MemoryItem.from_ai_action(ai_msg, result_msg)
        #     logger.debug(f"Storing the following memory:\n{memory_to_add.dump()}")
        #     memory_store.add(memory_to_add)

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
        message_sequence.add("system", budget_message)
        current_tokens_used += count_message_tokens(message_sequence[-1], model)

    # Append user input, the length of this is accounted for above
    message_sequence.append(user_input_msg)

    plugin_count = len(config.plugins)
    for i, plugin in enumerate(config.plugins):
        if not plugin.can_handle_on_planning():
            continue
        plugin_response = plugin.on_planning(
            agent.ai_config.prompt_generator, message_sequence.raw()
        )
        if not plugin_response or plugin_response == "":
            continue
        tokens_to_add = count_message_tokens(Message("system", plugin_response), model)
        if current_tokens_used + tokens_to_add > send_token_limit:
            logger.debug(f"Plugin response too long, skipping: {plugin_response}")
            logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
            break
        message_sequence.add("system", plugin_response)
        current_tokens_used += tokens_to_add

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
    for message in message_sequence:
        # Skip printing the prompt
        if message.role == "system" and message.content == system_prompt:
            continue
        logger.debug(f"{message.role.capitalize()}: {message.content}")
        logger.debug("")
    logger.debug("----------- END OF CONTEXT ----------------")
    agent.log_cycle_handler.log_cycle(
        agent.ai_name,
        agent.created_at,
        agent.cycle_count,
        message_sequence.raw(),
        CURRENT_CONTEXT_FILE_NAME,
    )

    # TODO: use a model defined elsewhere, so that model can contain
    # temperature and other settings we care about
    assistant_reply = create_chat_completion(
        prompt=message_sequence,
        config=agent.config,
        functions=openai_functions,
        max_tokens=tokens_remaining,
    )

    # Update full message history
    agent.history.append(user_input_msg)
    agent.history.add("assistant", assistant_reply.content, "ai_response")

    return assistant_reply
