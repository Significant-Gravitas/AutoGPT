import json
import random
import commands as cmd
import memory as mem
import data
import chat
from colorama import Fore, Style
from spinner import Spinner
import time
import speak
from enum import Enum, auto
import sys
from config import Config
from json_parser import fix_and_parse_json
from ai_config import AIConfig
import traceback
import yaml
import argparse


def print_to_console(
        title,
        title_color,
        content,
        speak_text=False,
        min_typing_speed=0.05,
        max_typing_speed=0.01):
    global cfg
    if speak_text and cfg.speak_mode:
        speak.say_text(f"{title}. {content}")
    print(title_color + title + " " + Style.RESET_ALL, end="")
    if content:
        if isinstance(content, list):
            content = " ".join(content)
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
    global cfg
    try:
        # Parse and print Assistant response
        assistant_reply_json = fix_and_parse_json(assistant_reply)

        # Check if assistant_reply_json is a string and attempt to parse it into a JSON object
        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = json.loads(assistant_reply_json)
            except json.JSONDecodeError as e:
                print_to_console("Error: Invalid JSON\n", Fore.RED, assistant_reply)
                assistant_reply_json = {}

        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None
        assistant_thoughts = assistant_reply_json.get("thoughts", {})
        assistant_thoughts_text = assistant_thoughts.get("text")

        if assistant_thoughts:
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
            assistant_thoughts_speak = assistant_thoughts.get("speak")

        print_to_console(f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text)
        print_to_console("REASONING:", Fore.YELLOW, assistant_thoughts_reasoning)

        if assistant_thoughts_plan:
            print_to_console("PLAN:", Fore.YELLOW, "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split('\n')
            for line in lines:
                line = line.lstrip("- ")
                print_to_console("- ", Fore.GREEN, line.strip())

        print_to_console("CRITICISM:", Fore.YELLOW, assistant_thoughts_criticism)
        # Speak the assistant's thoughts
        if cfg.speak_mode and assistant_thoughts_speak:
            speak.say_text(assistant_thoughts_speak)

    except json.decoder.JSONDecodeError:
        print_to_console("Error: Invalid JSON\n", Fore.RED, assistant_reply)

    # All other errors, return "Error: + error message"
    except Exception as e:
        call_stack = traceback.format_exc()
        print_to_console("Error: \n", Fore.RED, call_stack)


def load_variables(config_file="config.yaml"):
    # Load variables from yaml file if it exists
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ai_name = config.get("ai_name")
        ai_role = config.get("ai_role")
        ai_goals = config.get("ai_goals")
    except FileNotFoundError:
        ai_name = ""
        ai_role = ""
        ai_goals = []

    # Prompt the user for input if config file is missing or empty values
    if not ai_name:
        ai_name = input("Name your AI: ")
        if ai_name == "":
            ai_name = "Entrepreneur-GPT"

    if not ai_role:        
        ai_role = input(f"{ai_name} is: ")
        if ai_role == "":
            ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    if not ai_goals:
        print("Enter up to 5 goals for your AI: ")
        print("For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
        print("Enter nothing to load defaults, enter nothing when finished.")
        ai_goals = []
        for i in range(5):
            ai_goal = input(f"Goal {i+1}: ")
            if ai_goal == "":
                break
            ai_goals.append(ai_goal)
        if len(ai_goals) == 0:
            ai_goals = ["Increase net worth", "Grow Twitter Account", "Develop and manage multiple businesses autonomously"]
         
    # Save variables to yaml file
    config = {"ai_name": ai_name, "ai_role": ai_role, "ai_goals": ai_goals}
    with open(config_file, "w") as file:
        documents = yaml.dump(config, file)

    prompt = data.load_prompt()
    prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications."""

    # Construct full prompt
    full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
    for i, goal in enumerate(ai_goals):
        full_prompt += f"{i+1}. {goal}\n"

    full_prompt += f"\n\n{prompt}"
    return full_prompt


def construct_prompt():
    config = AIConfig.load()
    if config.ai_name:
        print_to_console(
            f"Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {config.ai_name}?",
            speak_text=True)
        should_continue = input(f"""Continue with the last settings? 
Name:  {config.ai_name}
Role:  {config.ai_role}
Goals: {config.ai_goals}  
Continue (y/n): """)
        if should_continue.lower() == "n":
            config = AIConfig()

    if not config.ai_name:         
        config = prompt_user()
        config.save()

    # Get rid of this global:
    global ai_name
    ai_name = config.ai_name
    
    full_prompt = config.construct_full_prompt()
    return full_prompt


def prompt_user():
    ai_name = ""
    # Construct the prompt
    print_to_console(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load defaults.",
        speak_text=True)

    # Get AI Name from User
    print_to_console(
        "Name your AI: ",
        Fore.GREEN,
        "For example, 'Entrepreneur-GPT'")
    ai_name = input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    print_to_console(
        f"{ai_name} here!",
        Fore.LIGHTBLUE_EX,
        "I am at your service.",
        speak_text=True)

    # Get AI Role from User
    print_to_console(
        "Describe your AI's role: ",
        Fore.GREEN,
        "For example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'")
    ai_role = input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    print_to_console(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
    print("Enter nothing to load defaults, enter nothing when finished.", flush=True)
    ai_goals = []
    for i in range(5):
        ai_goal = input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if len(ai_goals) == 0:
        ai_goals = ["Increase net worth", "Grow Twitter Account",
                    "Develop and manage multiple businesses autonomously"]

    config = AIConfig(ai_name, ai_role, ai_goals)
    return config

def parse_arguments():
    global cfg
    cfg.set_continuous_mode(False)
    cfg.set_speak_mode(False)
    
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--continuous', action='store_true', help='Enable Continuous Mode')
    parser.add_argument('--speak', action='store_true', help='Enable Speak Mode')
    parser.add_argument('--debug', action='store_true', help='Enable Debug Mode')
    parser.add_argument('--gpt3only', action='store_true', help='Enable GPT3.5 Only Mode')
    args = parser.parse_args()

    if args.continuous:
        print_to_console("Continuous Mode: ", Fore.RED, "ENABLED")
        print_to_console(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise. Use at your own risk.")
        cfg.set_continuous_mode(True)

    if args.speak:
        print_to_console("Speak Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_speak_mode(True)

    if args.gpt3only:
        print_to_console("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_smart_llm_model(cfg.fast_llm_model)


# TODO: fill in llm values here

cfg = Config()
parse_arguments()
ai_name = ""
prompt = construct_prompt()
# print(prompt)
# Initialize variables
full_message_history = []
result = None
# Make a constant:
user_input = "Determine which next command to use, and respond using the format specified above:"

# Interaction Loop
while True:
    # Send message to AI, get response
    with Spinner("Thinking... "):
        assistant_reply = chat.chat_with_ai(
            prompt,
            user_input,
            full_message_history,
            mem.permanent_memory,
            cfg.fast_token_limit) # TODO: This hardcodes the model to use GPT3.5. Make this an argument

    # print("assistant reply: "+assistant_reply)
    # Print Assistant thoughts
    print_assistant_thoughts(assistant_reply)

    # Get command name and arguments
    try:
        command_name, arguments = cmd.get_command(assistant_reply)
    except Exception as e:
        print_to_console("Error: \n", Fore.RED, str(e))

    if not cfg.continuous_mode:
        ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
        # Get key press: Prompt the user to press enter to continue or escape
        # to exit
        user_input = ""
        print_to_console(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")
        print(
            f"Enter 'y' to authorise command or 'n' to exit program, or enter feedback for {ai_name}...",
            flush=True)
        while True:
            console_input = input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
            if console_input.lower() == "y":
                user_input = "GENERATE NEXT COMMAND JSON"
                break
            elif console_input.lower() == "n":
                user_input = "EXIT"
                break
            else:
                user_input = console_input
                command_name = "human_feedback"
                break

        if user_input == "GENERATE NEXT COMMAND JSON":
            print_to_console(
            "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
            Fore.MAGENTA,
            "")
        elif user_input == "EXIT":
            print("Exiting...", flush=True)
            break
    else:
        # Print command
        print_to_console(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")

    # Execute command
    if command_name.lower() == "error":
        result = f"Command {command_name} threw the following error: " + arguments
    elif command_name == "human_feedback":
        result = f"Human feedback: {user_input}"
    else:
        result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"

    # Check if there's a result from the command append it to the message
    # history
    if result is not None:
        full_message_history.append(chat.create_chat_message("system", result))
        print_to_console("SYSTEM: ", Fore.YELLOW, result)
    else:
        full_message_history.append(
            chat.create_chat_message(
                "system", "Unable to execute command"))
        print_to_console("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

