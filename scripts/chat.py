import time
import openai
from config import Config
import token_counter
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


# call_count = 0  # Add a global variable to count the number of times the function is called

def generate_context(prompt, relevant_memory, full_message_history, model):
    # global call_count  # Access the global variable inside the function
    # call_count += 1

    # # Only include the initial messages for the first call
    # if call_count == 1:
    #     current_context = [
    #         create_chat_message("system", prompt),
    #         create_chat_message("system", f"The current time and date is {time.strftime('%c')}")
    #     ]
    # else:
    #     # Check if relevant_memory contains the specified string
    #     if "Please refer to the 'COMMANDS' list for availabe commands and only respond in the specified JSON format" in relevant_memory:
    #         current_context = [
    #             create_chat_message("system", prompt),
    #             create_chat_message("system", f"The current time and date is {time.strftime('%c')}")
    #         ]
    #     else:
    #         current_context = []
    current_context = [
        create_chat_message(
            "system", prompt),
        create_chat_message(
            "system", f"The current time and date is {time.strftime('%c')}"),
        create_chat_message(
            "system", f"This reminds you of these events from your past:\n{relevant_memory}\n\n")]

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    # Count the currently used tokens
    current_tokens_used = token_counter.count_message_tokens(current_context, model)
    return next_message_to_add_index, current_tokens_used, insertion_index, current_context


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

            if cfg.debug_mode:
                print(f"Token limit: {token_limit}")

            send_token_limit = token_limit - 1000

            relevant_memory = permanent_memory.get_relevant(str(full_message_history[-5:]), 10)

            if cfg.debug_mode:
                print('Memory Stats: ', permanent_memory.get_stats())

            next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                prompt, relevant_memory, full_message_history, model)

            while current_tokens_used > 2500:
                # remove memories until we are under 2500 tokens
                relevant_memory = relevant_memory[1:]
                next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                    prompt, relevant_memory, full_message_history, model)

            current_tokens_used += token_counter.count_message_tokens([create_chat_message("user", user_input)], model) # Account for user input (appended later)

            while next_message_to_add_index >= 0:
                # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = token_counter.count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                # Add the most recent message to the start of the current context, after the two system prompts.
                current_context.insert(insertion_index, full_message_history[next_message_to_add_index])

                # Count the currently used tokens
                current_tokens_used += tokens_to_add

                # Move to the next most recent message in the full message history
                next_message_to_add_index -= 1

            # Append user input, the length of this is accounted for above
            current_context.extend([create_chat_message("user", user_input)])

            # Calculate remaining tokens
            tokens_remaining = token_limit - current_tokens_used

            # TODO: use a model defined elsewhere, so that model can contain temperature and other settings we care about
            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )
            print(f"assistant_reply->>{assistant_reply}")
            # Update full message history
            full_message_history.append(
                create_chat_message(
                    "user", user_input))
            full_message_history.append(
                create_chat_message(
                    "assistant", assistant_reply))

            return assistant_reply
        except openai.error.RateLimitError:
            # TODO: WHen we switch to langchain, this is built in
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(2)
