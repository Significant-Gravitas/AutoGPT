import json
import commands as cmd
import memory as mem
import data
import chat
from colorama import Fore, Style
from spinner import Spinner

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
        
        print(Fore.YELLOW + "ASSISTANT THOUGHTS:" + Style.RESET_ALL, assistant_thoughts_text)
        print(Fore.YELLOW + "REASONING:" + Style.RESET_ALL, assistant_thoughts_reasoning)
        if assistant_thoughts_plan:
            print(Fore.YELLOW + "PLAN:" + Style.RESET_ALL)
            if assistant_thoughts_plan:
                # Split the input_string using the newline character and dash
                lines = assistant_thoughts_plan.split('\n- ')

                # Iterate through the lines and print each one with a bullet point
                for line in lines:
                    print(Fore.GREEN + "- " + Style.RESET_ALL + line.strip())
        print(Fore.YELLOW + "CRITICISM:" + Style.RESET_ALL, assistant_thoughts_criticism)

    except json.decoder.JSONDecodeError:
        print(Fore.RED + "Error: Invalid JSON" + Style.RESET_ALL)
        print(assistant_reply, flush=True)
    # All other errors, return "Error: + error message"
    except Exception as e:
        print(Fore.RED + "Error: " + str(e) + Style.RESET_ALL)
        print(assistant_reply, flush=True)

# Initialize variables
full_message_history = []
prompt = data.load_prompt()
token_limit = 6000  # The maximum number of tokens allowed in the API call
result = None
user_input = "NEXT COMMAND"

# Interaction Loop
while True:
    # Send message to AI, get response
    with Spinner("Thinking... "):
        assistant_reply = chat.chat_with_ai(prompt, user_input, full_message_history, mem.permanent_memory, token_limit)

    # Print Assistant thoughts
    print_assistant_thoughts(assistant_reply)

    # Get command name and arguments
    command_name, arguments = cmd.get_command(assistant_reply)
    
    ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
    # Get key press: Prompt the user to press enter to continue or escape to exit
    user_input = ""
    print(Fore.CYAN + f"NEXT ACTION: COMMAND = {command_name}  ARGUMENTS = {arguments}" + Style.RESET_ALL)
    print("Enter 'y' to authorise command or 'n' to exit program...", flush=True)
    while True:
        console_input = input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
        if console_input.lower() == "y":
            user_input = "NEXT COMMAND"
            break
        elif console_input.lower() == "n":
            user_input = "EXIT"
            break
        else:
            continue

    if user_input != "NEXT COMMAND":
        print("Exiting...", flush=True)
        break

    print(Fore.MAGENTA + "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=" + Style.RESET_ALL)

    # Exectute command
    result = cmd.execute_command(command_name, arguments)
    

    # Check if there's a result from the command append it to the message history
    if result != None:
        full_message_history.append(chat.create_chat_message("system", result))
        print(Fore.YELLOW + "SYSTEM: " + Style.RESET_ALL + result, flush=True)
    else:
        full_message_history.append(chat.create_chat_message("system", "Unable to execute command"))
        print(Fore.YELLOW + "SYSTEM: " + Style.RESET_ALL + "Unable to execute command", flush=True)
