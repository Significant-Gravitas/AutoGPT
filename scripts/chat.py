import time

import openai
import token_counter
from config import Config
from dotenv import load_dotenv
from llm_utils import create_chat_completion

cfg = Config()


def create_chat_message(role, content):
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
        create_chat_message(
            "system",
            f"This reminds you of these events from your past:\n{relevant_memory}\n\n",
        ),
    ]

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    # Count the currently used tokens
    current_tokens_used = token_counter.count_message_tokens(current_context, model)
    return (
        next_message_to_add_index,
        current_tokens_used,
        insertion_index,
        current_context,
    )


def chat_with_ai(
    prompt, user_input, full_message_history, permanent_memory, token_limit
):
    """Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory."""
    while True:
        try:
            model = cfg.fast_llm_model
            send_token_limit = token_limit - 1000

            relevant_memory = permanent_memory.get_relevant(
                str(full_message_history[-5:]), 10
            )

            (
                next_message_to_add_index,
                current_tokens_used,
                insertion_index,
                current_context,
            ) = generate_context(prompt, relevant_memory, full_message_history, model)

            while current_tokens_used > 2500:
                relevant_memory = relevant_memory[1:]
                (
                    next_message_to_add_index,
                    current_tokens_used,
                    insertion_index,
                    current_context,
                ) = generate_context(
                    prompt, relevant_memory, full_message_history, model
                )

            current_tokens_used += token_counter.count_message_tokens(
                [create_chat_message("user", user_input)], model
            )

            while next_message_to_add_index >= 0:
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = token_counter.count_message_tokens(
                    [message_to_add], model
                )
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                current_context.insert(
                    insertion_index, full_message_history[next_message_to_add_index]
                )
                current_tokens_used += tokens_to_add
                next_message_to_add_index -= 1

            current_context.extend([create_chat_message("user", user_input)])
            tokens_remaining = token_limit - current_tokens_used

            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            full_message_history.append(create_chat_message("user", user_input))
            full_message_history.append(
                create_chat_message("assistant", assistant_reply)
            )

            return assistant_reply
        except openai.error.RateLimitError:
            print("Error: API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)
