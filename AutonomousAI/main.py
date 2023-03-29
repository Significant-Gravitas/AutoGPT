import json
import commands as cmd
import memory as mem
import data
import chat
from colorama import Fore, Style
from spinner import Spinner

def print_to_console(title, title_color, content):
    print(title_color + title + Style.RESET_ALL, content, flush=True)

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
        
        print_to_console("ASSISTANT THOUGHTS:", Fore.YELLOW, assistant_thoughts_text)
        print_to_console("REASONING:", Fore.YELLOW, assistant_thoughts_reasoning)
        if assistant_thoughts_plan:
            print_to_console("Plan:", Fore.YELLOW, "")
            if assistant_thoughts_plan:
                # Split the input_string using the newline character and dash
                lines = assistant_thoughts_plan.split('\n- ')

                # Iterate through the lines and print each one with a bullet point
                for line in lines:
                    # Remove any "-" characters from the start of the line
                    line = line.lstrip("- ")
                    print_to_console("- ", Fore.GREEN, line.strip())
        print_to_console("CRITICISM:", Fore.YELLOW, assistant_thoughts_criticism)

    except json.decoder.JSONDecodeError:
        print_to_console("Error: Invalid JSON\n", Fore.RED, assistant_reply)
    # All other errors, return "Error: + error message"
    except Exception as e:
        print_to_console("Error: \n", Fore.RED, str(e))

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
    print_to_console("NEXT ACTION: ", Fore.CYAN, f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")
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

    print_to_console("-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=", Fore.MAGENTA, "")

    # Exectute command
    result = cmd.execute_command(command_name, arguments)
    

    # Check if there's a result from the command append it to the message history
    if result != None:
        full_message_history.append(chat.create_chat_message("system", result))
        print_to_console("SYSTEM: ", Fore.YELLOW, result)
    else:
        full_message_history.append(chat.create_chat_message("system", "Unable to execute command"))
        print_to_console("SYSTEM: ", Fore.YELLOW, "Unable to execute command")
