import time
import openai
from dotenv import load_dotenv
from config import Config
import token_counter
from llm_utils import create_chat_completion
from logger import logger
import logging

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

def reduce_dataset(dataset, limit=2500, f=len, *args, **kwargs):
    retval = []
    current_length = 0

    for i in dataset:
        if current_length + (x := f(i, *args, **kwargs)) <= limit:
            retval.append(i)
            current_length += x
        else:
            break

    return list(retval)

def generate_context(prompt, user_input, relevant_memory, full_message_history):
    current_context = [
        create_chat_message( "system", prompt),
        create_chat_message( "system", f"The current time and date is {time.strftime('%c')}"),
        create_chat_message( "system", f"This reminds you of these events from your past:\n{relevant_memory}\n\n"),
        *full_message_history,
        create_chat_message( "user", user_input)
        ]
    return current_context

# TODO: Change debug from hardcode to argument
def chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit):
    """Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory."""
    while True:
        try:
            """
            Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory.

            Args:
            prompt (str): The prompt explaining the rules to the AI.
            user_input (str): The input from the user.
            full_message_history (list): The list of all messages sent between the user and the AI.
            permanent_memory (Obj): The memory object containing the permanent memory.
            token_limit (int): The maximum number of tokens allowed in the API call.

            Returns:
            str: The AI's response.
            """
            model = cfg.fast_llm_model # TODO: Change model from hardcode to argument
            # Reserve 1000 tokens for the response

            logger.debug(f"Token limit: {token_limit}")
            send_token_limit = token_limit - 1000

            relevant_memory = '' if len(full_message_history) ==0 else  permanent_memory.get_relevant(str(full_message_history[-9:]), 10)

            logger.debug(f'Memory Stats: {permanent_memory.get_stats()}')

            # How much space is left in the token limit for memories?
            empty_context = generate_context(prompt, user_input, [], [])
            empty_context_tokens = token_counter.count_message_tokens(empty_context, model)
            memories_token_limit = 2500 - empty_context_tokens
            msg_history_token_limit = send_token_limit - empty_context_tokens - memories_token_limit

            # Reduce the memories to fit in the token limit
            reduced_memories = reduce_dataset(relevant_memory,
                                              limit=memories_token_limit,
                                              f=token_counter.count_string_tokens,
                                              model=model
                                              )

            # Reduce the message history to fit in the token limit
            reduced_message_history = reduce_dataset(full_message_history[::-1],
                                                     limit=msg_history_token_limit,
                                                     f=token_counter.count_message_tokens,
                                                     model=model
                                                     )[::-1]

            # Generate the context
            current_context = generate_context(prompt, user_input, reduced_memories, reduced_message_history)
            context_tokens = token_counter.count_message_tokens(current_context, model)

            tokens_remaining = token_limit - context_tokens


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

            # TODO: use a model defined elsewhere, so that model can contain temperature and other settings we care about
            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            # Update full message history
            full_message_history.append( create_chat_message( "user", user_input))
            full_message_history.append( create_chat_message( "assistant", assistant_reply))

            return assistant_reply
        except openai.error.RateLimitError:
            # TODO: When we switch to langchain, this is built in
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)
