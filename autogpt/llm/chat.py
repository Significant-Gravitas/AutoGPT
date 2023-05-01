import time
from random import shuffle

from openai.error import RateLimitError

from autogpt.config import Config
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import Message
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.llm.token_counter import count_message_tokens
from autogpt.logs import logger
from autogpt.memory_management.store_memory import (
    save_memory_trimmed_from_context_window,
)
from autogpt.memory_management.summary_memory import (
    get_newly_trimmed_messages,
    update_running_summary,
)

cfg = Config()


def create_chat_message(role, content) -> Message:
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}


def generate_context(prompt, relevant_memory, full_message_history, model):
    current_context = [
        create_chat_message("system", prompt),
        create_chat_message(
            "system", f"The current time and date is {time.strftime('%c')}"
        ),
        # create_chat_message(
        #     "system",
        #     f"This reminds you of these events from your past:\n{relevant_memory}\n\n",
        # ),
    ]

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    # Count the currently used tokens
    current_tokens_used = count_message_tokens(current_context, model)
    return (
        next_message_to_add_index,
        current_tokens_used,
        insertion_index,
        current_context,
    )


# TODO: Change debug from hardcode to argument
def chat_with_ai(
    agent, prompt, user_input, full_message_history, permanent_memory, token_limit
):
    """Interact with the OpenAI API, sending the prompt, user input, message history,
    and permanent memory."""
    while True:
        try:
            """
            Interact with the OpenAI API, sending the prompt, user input,
                message history, and permanent memory.

            Args:
                prompt (str): The prompt explaining the rules to the AI.
                user_input (str): The input from the user.
                full_message_history (list): The list of all messages sent between the
                    user and the AI.
                permanent_memory (Obj): The memory object containing the permanent
                  memory.
                token_limit (int): The maximum number of tokens allowed in the API call.

            Returns:
            str: The AI's response.
            """
            model = cfg.fast_llm_model  # TODO: Change model from hardcode to argument
            # Reserve 1000 tokens for the response
            logger.debug(f"Token limit: {token_limit}")
            send_token_limit = token_limit - 1000

            # if len(full_message_history) == 0:
            #     relevant_memory = ""
            # else:
            #     recent_history = full_message_history[-5:]
            #     shuffle(recent_history)
            #     relevant_memories = permanent_memory.get_relevant(
            #         str(recent_history), 5
            #     )
            #     if relevant_memories:
            #         shuffle(relevant_memories)
            #     relevant_memory = str(relevant_memories)
            relevant_memory = ""
            logger.debug(f"Memory Stats: {permanent_memory.get_stats()}")

            (
                next_message_to_add_index,
                current_tokens_used,
                insertion_index,
                current_context,
            ) = generate_context(prompt, relevant_memory, full_message_history, model)

            # while current_tokens_used > 2500:
            #     # remove memories until we are under 2500 tokens
            #     relevant_memory = relevant_memory[:-1]
            #     (
            #         next_message_to_add_index,
            #         current_tokens_used,
            #         insertion_index,
            #         current_context,
            #     ) = generate_context(
            #         prompt, relevant_memory, full_message_history, model
            #     )

            current_tokens_used += count_message_tokens(
                [create_chat_message("user", user_input)], model
            )  # Account for user input (appended later)

            current_tokens_used += 500  # Account for memory (appended later) TODO: The final memory may be less than 500 tokens

            # Add Messages until the token limit is reached or there are no more messages to add.
            while next_message_to_add_index >= 0:
                # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    # save_memory_trimmed_from_context_window(
                    #     full_message_history,
                    #     next_message_to_add_index,
                    #     permanent_memory,
                    # )
                    break

                # Add the most recent message to the start of the current context,
                #  after the two system prompts.
                current_context.insert(
                    insertion_index, full_message_history[next_message_to_add_index]
                )

                # Count the currently used tokens
                current_tokens_used += tokens_to_add

                # Move to the next most recent message in the full message history
                next_message_to_add_index -= 1

            # Insert Memories
            if len(full_message_history) > 0:
                (
                    newly_trimmed_messages,
                    agent.last_memory_index,
                ) = get_newly_trimmed_messages(
                    full_message_history=full_message_history,
                    current_context=current_context,
                    last_memory_index=agent.last_memory_index,
                )
                agent.summary_memory = update_running_summary(
                    current_memory=agent.summary_memory,
                    new_events=newly_trimmed_messages,
                )
                current_context.insert(insertion_index, agent.summary_memory)

            api_manager = ApiManager()
            # inform the AI about its remaining budget (if it has one)
            if api_manager.get_total_budget() > 0.0:
                remaining_budget = (
                    api_manager.get_total_budget() - api_manager.get_total_cost()
                )
                if remaining_budget < 0:
                    remaining_budget = 0
                system_message = (
                    f"Your remaining API budget is ${remaining_budget:.3f}"
                    + (
                        " BUDGET EXCEEDED! SHUT DOWN!\n\n"
                        if remaining_budget == 0
                        else " Budget very nearly exceeded! Shut down gracefully!\n\n"
                        if remaining_budget < 0.005
                        else " Budget nearly exceeded. Finish up.\n\n"
                        if remaining_budget < 0.01
                        else "\n\n"
                    )
                )
                logger.debug(system_message)
                current_context.append(create_chat_message("system", system_message))

            # Append user input, the length of this is accounted for above
            current_context.extend([create_chat_message("user", user_input)])

            plugin_count = len(cfg.plugins)
            for i, plugin in enumerate(cfg.plugins):
                if not plugin.can_handle_on_planning():
                    continue
                plugin_response = plugin.on_planning(
                    agent.prompt_generator, current_context
                )
                if not plugin_response or plugin_response == "":
                    continue
                tokens_to_add = count_message_tokens(
                    [create_chat_message("system", plugin_response)], model
                )
                if current_tokens_used + tokens_to_add > send_token_limit:
                    logger.debug("Plugin response too long, skipping:", plugin_response)
                    logger.debug("Plugins remaining at stop:", plugin_count - i)
                    break
                current_context.append(create_chat_message("system", plugin_response))

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
            for message in current_context:
                # Skip printing the prompt
                if message["role"] == "system" and message["content"] == prompt:
                    continue
                logger.debug(f"{message['role'].capitalize()}: {message['content']}")
                logger.debug("")
            logger.debug("----------- END OF CONTEXT ----------------")

            # TODO: use a model defined elsewhere, so that model can contain
            # temperature and other settings we care about
            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            # Update full message history
            full_message_history.append(create_chat_message("user", user_input))
            full_message_history.append(
                create_chat_message("assistant", assistant_reply)
            )

            return assistant_reply
        except RateLimitError:
            # TODO: When we switch to langchain, this is built in
            logger.warn("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)
