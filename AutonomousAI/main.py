import json
import random
import commands as cmd
import memory as mem
import data
import chat
from colorama import Fore, Style
from spinner import Spinner
import time

def print_to_console(title, title_color, content, min_typing_speed=0.05, max_typing_speed=0.01):
    print(title_color + title + " " + Style.RESET_ALL, end="")
    if content:
        words = content.split()
        for i, word in enumerate(words):
            print(word, end="", flush=True)
            if i < len(words) - 1:
                print(" ", end="", flush=True)
            typing_speed = random.uniform(min_typing_speed, max_typing_speed)
            time.sleep(typing_speed)
            # type faster after each word
            min_typing_speed = min_typing_speed * 0.95
            max_typing_speed = max_typing_speed * 0.95
    print()

def print_assistant_thoughts(assistant_reply):
    global ai_name
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
        
        print_to_console(f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text)
        print_to_console("REASONING:", Fore.YELLOW, assistant_thoughts_reasoning)
        if assistant_thoughts_plan:
            print_to_console("PLAN:", Fore.YELLOW, "")
            if assistant_thoughts_plan:

                # Split the input_string using the newline character and dash
                lines = assistant_thoughts_plan.split('\n')

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

def construct_prompt():
    global ai_name
    # Construct the prompt
    print_to_console("Welcome to Auto-GPT! ", Fore.GREEN, "Enter the name of your AI and its role below. Entering nothing will load defaults.")

    # Get AI Name from User
    print_to_console("Name your AI: ", Fore.GREEN, "For example, 'Entrepreneur-GPT'")
    ai_name = input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    # Get AI Role from User
    print_to_console("Describe your AI's role: ", Fore.GREEN, "For example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'")
    ai_role = input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    print_to_console("Enter up to 5 goals for your AI: ", Fore.GREEN, "For example: \nIncrease net worth \nGrow Twitter Account \nDevelop and manage multiple businesses autonomously'")
    print("Enter nothing to load defaults, enter nothing when finished.", flush=True)
    ai_goals = []
    for i in range(5):
        ai_goal = input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if len(ai_goals) == 0:
        ai_goals = ["Increase net worth", "Grow Twitter Account", "Develop and manage multiple businesses autonomously"]

    prompt = data.load_prompt()
    prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications."""

    # Construct full prompt
    full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
    for i, goal in enumerate(ai_goals):
        full_prompt += f"{i+1}. {goal}\n"

    full_prompt += f"\n\n{prompt}"
    return full_prompt

# Check if the python script was executed with arguments, get those arguments
continuous_mode = False
import sys
if len(sys.argv) > 1:
    # Check if the first argument is "continuous-mode"
    if sys.argv[1] == "continuous-mode":
        # Set continuous mode to true
        print_to_console("Continuous Mode: ", Fore.RED, "ENABLED")
        # print warning
        print_to_console("WARNING: ", Fore.RED, "Continuous mode is not recommended. It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise. Use at your own risk.")
        continuous_mode = True

ai_name = ""
prompt = construct_prompt()
# Initialize variables
full_message_history = []
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
    try:
        command_name, arguments = cmd.get_command(assistant_reply)
    except Exception as e:
        print_to_console("Error: \n", Fore.RED, str(e))
        

    if not continuous_mode:
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
    else:
        # Print command
        print_to_console("NEXT ACTION: ", Fore.CYAN, f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")

    # Exectute command
    if command_name.lower() != "error":
        result = cmd.execute_command(command_name, arguments)
    else:
        result ="Error: " + arguments

    # Check if there's a result from the command append it to the message history
    if result != None:
        full_message_history.append(chat.create_chat_message("system", result))
        print_to_console("SYSTEM: ", Fore.YELLOW, result)
    else:
        full_message_history.append(chat.create_chat_message("system", "Unable to execute command"))
        print_to_console("SYSTEM: ", Fore.YELLOW, "Unable to execute command")
