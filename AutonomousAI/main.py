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

def chat_with_ai(prompt, user_input, full_message_history, permanent_memory, token_limit, debug = False):
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
    if debug:
        print("------------ CONTEXT SENT TO AI ---------------")
        for message in current_context:
            # Skip printing the prompt
            if message["role"] == "system" and message["content"] == prompt:
                continue
            print(f"{message['role'].capitalize()}: {message['content']}")
        print("----------- END OF CONTEXT ----------------")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=current_context,
    )

    assistant_reply = response.choices[0].message["content"]

    # Update full message history
    full_message_history.append(create_chat_message("user", user_input))
    full_message_history.append(create_chat_message("assistant", assistant_reply))

    return assistant_reply

def get_command(response):
    try:
        response_json = json.loads(response)
        command = response_json["command"]
        command_name = command["name"]
        arguments = command["args"]

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error: Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)

def execute_command(command_name, arguments):
    try:
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
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)

def load_prompt():
    try:
        # Load the promt from data/prompt.txt
        with open("data/prompt.txt", "r") as prompt_file:
            prompt = prompt_file.read()

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""

def print_assistant_thoughts(assistant_reply):
    try:
        # Parse and print Assistant response
        assistant_reply_json = json.loads(assistant_reply)

        assistant_thoughts = assistant_reply_json.get("thoughts")
        if assistant_thoughts:
            assistant_thoughts_text = assistant_thoughts.get("text")
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
        else:
            assistant_thoughts_text = None
            assistant_thoughts_reasoning = None
            assistant_thoughts_plan = None
            assistant_thoughts_criticism = None
        
        print(f"ASSISTANT THOUGHTS: {assistant_thoughts_text}", flush=True)
        print(f"REASONING: {assistant_thoughts_reasoning}", flush=True)
        if assistant_thoughts_plan:
            print("PLAN: ", flush=True)
            if assistant_thoughts_plan:
                # Split the input_string using the newline character and dash
                lines = assistant_thoughts_plan.split('\n- ')

                # Iterate through the lines and print each one with a bullet point
                for line in lines:
                    print(f"- {line.strip()}", flush=True)
        print(f"CRITICISM: " + assistant_thoughts_criticism, flush=True)

    except json.decoder.JSONDecodeError:
        print("Error: Invalid JSON", flush=True)
        print(assistant_reply, flush=True)
    # All other errors, return "Error: + error message"
    except Exception as e:
        print("Error: " + str(e), flush=True)
        print(assistant_reply, flush=True)

# Initialize variables
full_message_history = []
prompt = load_prompt()
token_limit = 6000  # The maximum number of tokens allowed in the API call
result = None
user_input = "NEXT COMMAND"

# Interaction Loop
while True:
    # Send message to AI, get response
    assistant_reply = chat_with_ai(prompt, user_input, full_message_history, mem.permanent_memory, token_limit)

    # Print Assistant thoughts
    print_assistant_thoughts(assistant_reply)

    # Get command name and arguments
    command_name, arguments = get_command(assistant_reply)
    
    ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
    # Get key press: Prompt the user to press enter to continue or escape to exit
    user_input = ""
    print(f"""NEXT ACTION: COMMAND = {command_name}  ARGUMENTS = {arguments}""", flush=True)
    print("Enter 'y' to authorise command or 'n' to exit program...", flush=True)
    while True:
        console_input = input()
        if console_input == "y":
            user_input = "NEXT COMMAND"
            break
        elif console_input == "n":
            user_input = "EXIT"
            break
        else:
            continue

    if user_input != "NEXT COMMAND":
        print("Exiting...", flush=True)
        break

    print("-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=", flush=True)

    # Exectute command
    result = execute_command(command_name, arguments)
    

    # Check if there's a result from the command append it to the message history
    if result != None:
        full_message_history.append(create_chat_message("system", result))
        print("system: " + result, flush=True)
    else:
        full_message_history.append(create_chat_message("system", "Unable to execute command"))
        print("system: Unable to execute command", flush=True)
