import time
import openai
import keys

# Initialize the OpenAI API client
openai.api_key = keys.OPENAI_API_KEY


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


def chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit,
        debug=False):
    while True:
        try:
            """
            Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory.

            Args:
            prompt (str): The prompt explaining the rules to the AI.
            user_input (str): The input from the user.
            full_message_history (list): The list of all messages sent between the user and the AI.
            permanent_memory (list): The list of items in the AI's permanent memory.
            token_limit (int): The maximum number of tokens allowed in the API call.

            Returns:
            str: The AI's response.
            """
            current_context = [
                create_chat_message(
                    "system", prompt), create_chat_message(
                    "system", f"Permanent memory: {permanent_memory}")]
            current_context.extend(
                full_message_history[-(token_limit - len(prompt) - len(permanent_memory) - 10):])
            current_context.extend([create_chat_message("user", user_input)])

            # Debug print the current context
            if debug:
                print("------------ CONTEXT SENT TO AI ---------------")
                for message in current_context:
                    # Skip printing the prompt
                    if message["role"] == "system" and message["content"] == prompt:
                        continue
                    print(
                        f"{message['role'].capitalize()}: {message['content']}")
                print("----------- END OF CONTEXT ----------------")

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=current_context,
            )

            assistant_reply = response.choices[0].message["content"]

            # Update full message history
            full_message_history.append(
                create_chat_message(
                    "user", user_input))
            full_message_history.append(
                create_chat_message(
                    "assistant", assistant_reply))

            return assistant_reply
        except openai.error.RateLimitError:
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)
