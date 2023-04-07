import json
import random
from data.response_prompt import parse, Thought
from colorama import Fore, Style
import time
import speak
from json_parser import fix_and_parse_json
from ai_config import AIConfig
import traceback


def print_to_console(
    title,
    title_color,
    content,
    speak_text=False,
    min_typing_speed=0.05,
    max_typing_speed=0.01,
):
    if speak_text:
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


def print_assistant_thoughts(ai_name, assistant_reply, speak_text=False):
    try:
        # Parse and print Assistant response
        try:
            assistant_thoughts = parse(assistant_reply, Thought)
        except Exception as e:
            assistant_thoughts = None

        assistant_thoughts_text = None
        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None

        if assistant_thoughts:
            assistant_thoughts_text = assistant_thoughts.text
            assistant_thoughts_reasoning = assistant_thoughts.reasoning
            assistant_thoughts_plan = assistant_thoughts.plan
            assistant_thoughts_criticism = assistant_thoughts.criticism
            assistant_thoughts_speak = assistant_thoughts.speak

        print_to_console(
            f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text
        )
        print_to_console("REASONING:", Fore.YELLOW, assistant_thoughts_reasoning)

        if assistant_thoughts_plan:
            print_to_console("PLAN:", Fore.YELLOW, "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split("\n")
            for line in lines:
                line = line.lstrip("- ")
                print_to_console("- ", Fore.GREEN, line.strip())

        print_to_console("CRITICISM:", Fore.YELLOW, assistant_thoughts_criticism)
        # Speak the assistant's thoughts
        if speak_text and assistant_thoughts_speak:
            speak.say_text(assistant_thoughts_speak)

    except json.decoder.JSONDecodeError:
        print_to_console("Error: Invalid JSON\n", Fore.RED, assistant_reply)

    # All other errors, return "Error: + error message"
    except Exception as e:
        call_stack = traceback.format_exc()
        print_to_console("Error: " + str(e) + "\n", Fore.RED, call_stack)


def get_ai_config(should_speak=False):
    stored_ai_config = AIConfig.load()
    if stored_ai_config:
        print_to_console(
            "Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {stored_ai_config.ai_name}?",
            speak_text=should_speak,
        )
        should_continue = input(
            f"""Continue with the last settings? 
Name:  {stored_ai_config.ai_name}
Role:  {stored_ai_config.ai_role}
Goals: {stored_ai_config.ai_goals}  
Continue (y/n): """
        )
        if should_continue.lower() == "y":
            return stored_ai_config
    new_ai_config = AIConfig()
    new_ai_config = get_ai_config_from_user(should_speak)
    new_ai_config.save()
    return new_ai_config


def get_ai_config_from_user(should_speak=False):
    ai_name = ""
    # Construct the prompt
    print_to_console(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load defaults.",
        speak_text=should_speak,
    )

    # Get AI Name from User
    print_to_console("Name your AI: ", Fore.GREEN, "For example, 'Entrepreneur-GPT'")
    ai_name = input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    print_to_console(
        f"{ai_name} here!",
        Fore.LIGHTBLUE_EX,
        "I am at your service.",
        speak_text=should_speak,
    )

    # Get AI Role from User
    print_to_console(
        "Describe your AI's role: ",
        Fore.GREEN,
        "For example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'",
    )
    ai_role = input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    print_to_console(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'",
    )
    print(
        "Enter nothing to load defaults, enter nothing when finished.",
        flush=True,
    )
    ai_goals = []
    for i in range(5):
        ai_goal = input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if len(ai_goals) == 0:
        ai_goals = [
            "Increase net worth",
            "Grow Twitter Account",
            "Develop and manage multiple businesses autonomously",
        ]

    config = AIConfig(ai_name, ai_role, ai_goals)
    return config
