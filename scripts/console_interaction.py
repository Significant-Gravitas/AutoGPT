import json
import random
import time
import traceback
import speak

from colorama import Fore, Style
from config import Config
from json_parser import fix_and_parse_json
from ai_config import AIConfig

cfg = Config()

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


def print_assistant_thoughts(ai_name, assistant_reply):
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