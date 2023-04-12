import json
import random
import commands as cmd
import utils
from memory import get_memory
import data
import chat
from colorama import Fore, Style
from spinner import Spinner
import time
import speak
from config import Config
from json_parser import fix_and_parse_json
from ai_config import AIConfig
import traceback
import yaml
import argparse
from logger import logger
import logging

cfg = Config()

def check_openai_api_key():
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not cfg.openai_api_key:
        print(
            Fore.RED +
            "Please set your OpenAI API key in config.py or as an environment variable."
        )
        print("You can get your key from https://beta.openai.com/account/api-keys")
        exit(1)

def print_assistant_thoughts(assistant_reply):
    """Prints the assistant's thoughts to the console"""
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
                logger.error("Error: Invalid JSON\n", Fore.RED, assistant_reply)
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

        logger.typewriter_log(f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text)
        logger.typewriter_log("REASONING:", Fore.YELLOW, assistant_thoughts_reasoning)

        if assistant_thoughts_plan:
            logger.typewriter_log("PLAN:", Fore.YELLOW, "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split('\n')
            for line in lines:
                line = line.lstrip("- ")
                logger.typewriter_log("- ", Fore.GREEN, line.strip())

        logger.typewriter_log("CRITICISM:", Fore.YELLOW, assistant_thoughts_criticism)
        # Speak the assistant's thoughts
        if cfg.speak_mode and assistant_thoughts_speak:
            speak.say_text(assistant_thoughts_speak)

    except json.decoder.JSONDecodeError:
        logger.error("Error: Invalid JSON\n", Fore.RED, assistant_reply)

    # All other errors, return "Error: + error message"
    except Exception as e:
        call_stack = traceback.format_exc()
        logger.error("Error: \n", Fore.RED, call_stack)


def load_variables(config_file="config.yaml"):
    """Load variables from yaml file if it exists, otherwise prompt the user for input"""
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
        ai_name = utils.clean_input("Name your AI: ")
        if ai_name == "":
            ai_name = "Entrepreneur-GPT"

    if not ai_role:
        ai_role = utils.clean_input(f"{ai_name} is: ")
        if ai_role == "":
            ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    if not ai_goals:
        print("Enter up to 5 goals for your AI: ")
        print("For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
        print("Enter nothing to load defaults, enter nothing when finished.")
        ai_goals = []
        for i in range(5):
            ai_goal = utils.clean_input(f"Goal {i+1}: ")
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
    prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as a LLM and pursue simple strategies with no legal complications."""

    # Construct full prompt
    full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
    for i, goal in enumerate(ai_goals):
        full_prompt += f"{i+1}. {goal}\n"

    full_prompt += f"\n\n{prompt}"
    return full_prompt


def construct_prompt():
    """Construct the prompt for the AI to respond to"""
    config = AIConfig.load()
    if config.ai_name:
        logger.typewriter_log(
            f"Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {config.ai_name}?",
            speak_text=True)
        should_continue = utils.clean_input(f"""Continue with the last settings?
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
    """Prompt the user for input"""
    ai_name = ""
    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load defaults.",
        speak_text=True)

    # Get AI Name from User
    logger.typewriter_log(
        "Name your AI: ",
        Fore.GREEN,
        "For example, 'Entrepreneur-GPT'")
    ai_name = utils.clean_input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    logger.typewriter_log(
        f"{ai_name} here!",
        Fore.LIGHTBLUE_EX,
        "I am at your service.",
        speak_text=True)

    # Get AI Role from User
    logger.typewriter_log(
        "Describe your AI's role: ",
        Fore.GREEN,
        "For example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'")
    ai_role = utils.clean_input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
    print("Enter nothing to load defaults, enter nothing when finished.", flush=True)
    ai_goals = []
    for i in range(5):
        ai_goal = utils.clean_input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if len(ai_goals) == 0:
        ai_goals = ["Increase net worth", "Grow Twitter Account",
                    "Develop and manage multiple businesses autonomously"]

    config = AIConfig(ai_name, ai_role, ai_goals)
    return config

def parse_arguments():
    """Parses the arguments passed to the script"""
    global cfg
    cfg.set_debug_mode(False)
    cfg.set_continuous_mode(False)
    cfg.set_speak_mode(False)

    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--continuous', action='store_true', help='Enable Continuous Mode')
    parser.add_argument('--speak', action='store_true', help='Enable Speak Mode')
    parser.add_argument('--debug', action='store_true', help='Enable Debug Mode')
    parser.add_argument('--gpt3only', action='store_true', help='Enable GPT3.5 Only Mode')
    parser.add_argument('--gpt4only', action='store_true', help='Enable GPT4 Only Mode')
    args = parser.parse_args()

    if args.debug:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_debug_mode(True)

    if args.continuous:
        logger.typewriter_log("Continuous Mode: ", Fore.RED, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise. Use at your own risk.")
        cfg.set_continuous_mode(True)

    if args.speak:
        logger.typewriter_log("Speak Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_speak_mode(True)

    if args.gpt3only:
        logger.typewriter_log("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_smart_llm_model(cfg.fast_llm_model)

    if args.gpt4only:
        logger.typewriter_log("GPT4 Only Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_fast_llm_model(cfg.smart_llm_model)

    if args.debug:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_debug_mode(True)


# TODO: fill in llm values here
check_openai_api_key()
cfg = Config()
parse_arguments()
logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
ai_name = ""
prompt = construct_prompt()
# print(prompt)
# Initialize variables
full_message_history = []
result = None
next_action_count = 0
# Make a constant:
user_input = "Determine which next command to use, and respond using the format specified above:"

# Initialize memory and make sure it is empty.
# this is particularly important for indexing and referencing pinecone memory
memory = get_memory(cfg, init=True)
print('Using memory of type: ' + memory.__class__.__name__)

# Interaction Loop
while True:
    # Send message to AI, get response
    with Spinner("Thinking... "):
        assistant_reply = chat.chat_with_ai(
            prompt,
            user_input,
            full_message_history,
            memory,
            cfg.fast_token_limit) # TODO: This hardcodes the model to use GPT3.5. Make this an argument

    # Print Assistant thoughts
    print_assistant_thoughts(assistant_reply)

    # Get command name and arguments
    try:
        command_name, arguments = cmd.get_command(assistant_reply)
    except Exception as e:
        logger.error("Error: \n", Fore.RED, str(e))

    if not cfg.continuous_mode and next_action_count == 0:
        ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
        # Get key press: Prompt the user to press enter to continue or escape
        # to exit
        user_input = ""
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")
        print(
            f"Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for {ai_name}...",
            flush=True)
        while True:
            console_input = utils.clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
            if console_input.lower() == "y":
                user_input = "GENERATE NEXT COMMAND JSON"
                break
            elif console_input.lower().startswith("y -"):
                try:
                    next_action_count = abs(int(console_input.split(" ")[1]))
                    user_input = "GENERATE NEXT COMMAND JSON"
                except ValueError:
                    print("Invalid input format. Please enter 'y -n' where n is the number of continuous tasks.")
                    continue
                break
            elif console_input.lower() == "n":
                user_input = "EXIT"
                break
            else:
                user_input = console_input
                command_name = "human_feedback"
                break

        if user_input == "GENERATE NEXT COMMAND JSON":
            logger.typewriter_log(
            "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
            Fore.MAGENTA,
            "")
        elif user_input == "EXIT":
            print("Exiting...", flush=True)
            break
    else:
        # Print command
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")

    # Execute command
    if command_name is not None and command_name.lower().startswith( "error" ):
        result = f"Command {command_name} threw the following error: " + arguments
    elif command_name == "human_feedback":
        result = f"Human feedback: {user_input}"
    else:
        result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"
        if next_action_count > 0:
            next_action_count -= 1

    memory_to_add = f"Assistant Reply: {assistant_reply} " \
                    f"\nResult: {result} " \
                    f"\nHuman Feedback: {user_input} "

    memory.add(memory_to_add)

    # Check if there's a result from the command append it to the message
    # history
    if result is not None:
        full_message_history.append(chat.create_chat_message("system", result))
        logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
    else:
        full_message_history.append(
            chat.create_chat_message(
                "system", "Unable to execute command"))
        logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")
