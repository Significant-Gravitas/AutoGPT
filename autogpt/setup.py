"""Set up the AI and its goals"""
import os
from colorama import Fore, Style

from autogpt import utils
from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger


def prompt_user() -> AIConfig:
    """Prompt the user for input

    Returns:
        AIConfig: The AIConfig object containing the user's input
    """
    ai_name = ""
    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "run with '--help' for more information.",
        speak_text=True,
    )

    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load"
        " defaults.",
        speak_text=True,
    )

    # Get AI Name from User
    default_ai_name = os.getenv("DEFAULT_NAME", "Entrepreneur-GPT")
    logger.typewriter_log(
        "Name your AI: ", Fore.GREEN, f"For example, '{default_ai_name}'"
    )
    ai_name = utils.clean_input("AI Name: ")
    if ai_name == "":
        ai_name = default_ai_name

    logger.typewriter_log(
        f"{ai_name} here!", Fore.LIGHTBLUE_EX, "I am at your service.", speak_text=True
    )

    # Get AI Role from User
    default_ai_role = os.getenv("DEFAULT_ROLE", 
                            "an AI designed to autonomously develop and run businesses with the"
                            " sole goal of increasing your net worth.")
    logger.typewriter_log(
        "Describe your AI's role: ",
        Fore.GREEN,
        f"For example, '{default_ai_role}.'",
    )
    
    ai_role = utils.clean_input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = default_ai_role

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage"
        " multiple businesses autonomously'",
    )
    print("Enter nothing to load defaults, enter nothing when finished.", flush=True)
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

    return AIConfig(ai_name, ai_role, ai_goals)
