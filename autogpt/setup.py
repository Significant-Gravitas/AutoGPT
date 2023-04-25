"""Set up the AI and its goals"""
from colorama import Fore, Style

from autogpt import utils
from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger


def prompt_user(other: AIConfig = None) -> AIConfig:
    """Prompt the user for input

    Params:
        other (AIConfig): The AIConfig object to use as a template

    Returns:
        AIConfig: The AIConfig object containing the user's input
    """
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

    if other and other.ai_name:
        ai_name = other.ai_name
    else:
        ai_name = ""
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

    if other and other.ai_role:
        ai_role = other.ai_role
    else:
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

    if other and other.ai_goals:
        ai_goals = other.ai_goals
    else:
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
