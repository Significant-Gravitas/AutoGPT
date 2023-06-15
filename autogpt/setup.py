"""Set up the AI and its goals"""
import re

from colorama import Fore, Style
from jinja2 import Template

from autogpt import utils
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.llm.base import ChatSequence, Message
from autogpt.llm.chat import create_chat_completion
from autogpt.logs import logger
from autogpt.prompts.default_prompts import (
    DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_USER_DESIRE_PROMPT,
)

CFG = Config()


def prompt_user() -> AIConfig:
    """Prompt the user for input

    Returns:
        AIConfig: The AIConfig object tailored to the user's input
    """
    ai_name = ""
    ai_config = None

    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "run with '--help' for more information.",
        speak_text=True,
    )

    # Get user desire
    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        "input '--manual' to enter manual mode.",
        speak_text=True,
    )

    user_desire = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}I want Auto-GPT to{Style.RESET_ALL}: "
    )

    if user_desire == "":
        user_desire = DEFAULT_USER_DESIRE_PROMPT  # Default prompt

    # If user desire contains "--manual"
    if "--manual" in user_desire:
        logger.typewriter_log(
            "Manual Mode Selected",
            Fore.GREEN,
            speak_text=True,
        )
        return generate_aiconfig_manual()

    else:
        try:
            return generate_aiconfig_automatic(user_desire)
        except Exception as e:
            logger.typewriter_log(
                "Unable to automatically generate AI Config based on user desire.",
                Fore.RED,
                "Falling back to manual mode.",
                speak_text=True,
            )

            return generate_aiconfig_manual()


def generate_aiconfig_manual() -> AIConfig:
    """
    Interactively create an AI configuration by prompting the user to provide the name, role, and goals of the AI.

    This function guides the user through a series of prompts to collect the necessary information to create
    an AIConfig object. The user will be asked to provide a name and role for the AI, as well as up to five
    goals. If the user does not provide a value for any of the fields, default values will be used.

    Returns:
        AIConfig: An AIConfig object containing the user-defined or default AI name, role, and goals.
    """

    # Manual Setup Intro
    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load"
        " defaults.",
        speak_text=True,
    )

    # Get AI Name from User
    logger.typewriter_log(
        "Name your AI: ", Fore.GREEN, "For example, 'Entrepreneur-GPT'"
    )
    ai_name = utils.clean_input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    logger.typewriter_log(
        f"{ai_name} here!", Fore.LIGHTBLUE_EX, "I am at your service.", speak_text=True
    )

    # Get AI Role from User
    logger.typewriter_log(
        "Describe your AI's role: ",
        Fore.GREEN,
        "For example, 'an AI designed to autonomously develop and run businesses with"
        " the sole goal of increasing your net worth.'",
    )
    ai_role = utils.clean_input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the"
        " sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage"
        " multiple businesses autonomously'",
    )
    logger.info("Enter nothing to load defaults, enter nothing when finished.")
    ai_goals = []
    for i in range(5):
        ai_goal = utils.clean_input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if not ai_goals:
        ai_goals = [
            "Increase net worth",
            "Grow Twitter Account",
            "Develop and manage multiple businesses autonomously",
        ]

    # Get API Budget from User
    logger.typewriter_log(
        "Enter your budget for API calls: ",
        Fore.GREEN,
        "For example: $1.50",
    )
    logger.info("Enter nothing to let the AI run without monetary limit")
    api_budget_input = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}Budget{Style.RESET_ALL}: $"
    )
    if api_budget_input == "":
        api_budget = 0.0
    else:
        try:
            api_budget = float(api_budget_input.replace("$", ""))
        except ValueError:
            logger.typewriter_log(
                "Invalid budget input. Setting budget to unlimited.", Fore.RED
            )
            api_budget = 0.0

    return AIConfig(ai_name, ai_role, ai_goals, api_budget)


def generate_aiconfig_automatic(user_prompt) -> AIConfig:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """

    system_prompt = DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC
    prompt_ai_config_automatic = Template(
        DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC
    ).render(user_prompt=user_prompt)
    # Call LLM with the string as user input
    output = create_chat_completion(
        ChatSequence.for_model(
            CFG.fast_llm_model,
            [
                Message("system", system_prompt),
                Message("user", prompt_ai_config_automatic),
            ],
        )
    )

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # Parse the output
    ai_name = re.search(r"Name(?:\s*):(?:\s*)(.*)", output, re.IGNORECASE).group(1)
    ai_role = (
        re.search(
            r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        .group(1)
        .strip()
    )
    ai_goals = re.findall(r"(?<=\n)-\s*(.*)", output)
    api_budget = 0.0  # TODO: parse api budget using a regular expression

    return AIConfig(ai_name, ai_role, ai_goals, api_budget)
