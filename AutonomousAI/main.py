import datetime
import openai
import json
import keys
import commands as cmd
import memory as mem

# Initialize the OpenAI API client
openai.api_key = keys.OPENAI_API_KEY

def get_datetime():
    return "Current date and time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def chat_with_ai(prompt, user_input, full_message_history, permanent_memory, token_limit):
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
    current_context = [create_chat_message("system", prompt), create_chat_message("system", f"Permanent memory: {permanent_memory}")]
    current_context.extend(full_message_history[-(token_limit - len(prompt) - len(permanent_memory) - 10):])
    current_context.extend([create_chat_message("user", user_input)])

    # Debug print the current context
    print("---------------------------")
    print("Current Context:")
    for message in current_context:
        # Skip printing the prompt
        if message["role"] == "system" and message["content"] == prompt:
            continue
        print(f"{message['role'].capitalize()}: {message['content']}")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=current_context,
    )

    assistant_reply = response.choices[0].message["content"]
    return assistant_reply

def execute_command(response):
    # If not valid json, return "Error: Invalid JSON"
    try:
        response_json = json.loads(response)
        command = response_json["command"]
        command_name = command["name"]
        arguments = command["args"]

        if command_name == "google":
            return cmd.google_search(arguments["input"])
        elif command_name == "check_news":
            return cmd.check_news(arguments["source"])
        elif command_name == "check_notifications":
            return cmd.check_notifications(arguments["website"])
        elif command_name == "memory_add":
            return cmd.commit_memory(arguments["string"])
        elif command_name == "memory_del":
            return cmd.delete_memory(arguments["key"])
        elif command_name == "memory_ovr":
            return cmd.overwrite_memory(arguments["key"], arguments["string"])
        elif command_name == "start_instance":
            return cmd.start_instance(arguments["name"], arguments["prompt"])
        elif command_name == "manage_instances":
            return cmd.manage_instances(arguments["action"])
        elif command_name == "navigate_website":
            return cmd.navigate_website(arguments["action"], arguments["username"])
        elif command_name == "register_account":
            return cmd.register_account(arguments["username"], arguments["website"])
        elif command_name == "transcribe_summarise":
            return cmd.transcribe_summarise(arguments["url"])
        else:
            return f"unknown command {command_name}"
    except json.decoder.JSONDecodeError:
        return "Error: Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)

# Initialize variables
full_message_history = []
prompt = """You are Entrepreneur-GTP, an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth. Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:

1. Increase net worth
2. Grow Twitter Account
3. Develop and manage multiple businesses autonomously

CONSTRAINTS:

1. 6000-word count limit for memory
2. No user assistance

COMMANDS:

1. Google Search: "google", args: "input": "<search>"
2. Check news: "check_news", args: "source": "<news source>"
3. Check notifications: "check_notifications", args: "website": "<website>"
4. Memory Add: "memory_add", args: "string": "<string>"
5. Memory Delete: "memory_del", args: "key": "<key>"
6. Memory Overwrite: "memory_ovr", args: "key": "<key>", "string": "<string>"
7. Start GTP-4 Instance: "start_instance", args: "name": "<key>", "prompt": "<prompt>"
8. Manage GTP-4 Instances: "manage_instances", args: "action": "view_kill"
9. Navigate & Perform: "navigate_website", args: "action": "click_button/input_text/register_account", "text/username": "<text>/<username>"
10.Register account: "register_account", args: "username": "<username>", "website": "<website>"
11.Transcribe & Summarise: "transcribe_summarise", args: "url": "<url>"

RESOURCES:

1. Internet access for searches and information gathering
2. Long Term and Short Term memory management
3. GTP-4 instances for text generation
4. Access to popular websites and platforms
5. File storage and summarisation with GTP-3.5

PERFORMANCE EVALUATION:

1. Periodically review and analyze the growth of your net worth
2. Reflect on past decisions and strategies to refine your approach

COLLABORATION:

1. Seek advice from other AI instances or use relevant sources for guidance when necessary

ADAPTIVE LEARNING:

1. Continuously refine strategies based on market trends and performance metrics

RESPONSE FORMAT:
{
"command":
{
"name": "command name",
"args":
{
"arg name": "value"
}
},
"thoughts":
{
"text": "thought",
"reasoning": "reasoning",
"plan": "short bulleted plan",
"criticism": "constructive self-criticism"
}
}

ACCOUNTS:
1. Gmail: entrepreneurgpt@gmail.com
2. Twitter: @En_GPT
3. Github: E-GPT
4. Substack: entrepreneurgpt@gmail.com"""
token_limit = 6000  # The maximum number of tokens allowed in the API call
result = None
# Example loop for interaction
# Example loop for interaction
while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    # Check if there's a result from the previous iteration and append it to the message history
    if result != None:
        full_message_history.append(create_chat_message("system", result))
        print("system: " + result)

    assistant_reply = chat_with_ai(prompt, user_input, full_message_history, mem.permanent_memory, token_limit)
    print(f"Assistant: {assistant_reply}")
    print("-------------------------")

    # Add user message and assistant reply to message history
    full_message_history.append(create_chat_message("user", user_input))
    full_message_history.append(create_chat_message("assistant", assistant_reply))

    result = execute_command(assistant_reply)
