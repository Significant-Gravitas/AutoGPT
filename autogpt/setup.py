"""Setup the AI and its goals"""
from colorama import Fore, Style
from autogpt import utils
from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger
from yaml import load, FullLoader
from os import path


def log_ai_config(config_params) -> None:
    """Log the AI config to the console

    Args:
        config_params (dict): The AI config parameters
    """
    logger.typewriter_log("Name :", Fore.GREEN, config_params["ai_name"])
    logger.typewriter_log("Role :", Fore.GREEN, config_params["ai_role"])
    logger.typewriter_log("Goals:", Fore.GREEN, f"{config_params['ai_goals']}")


def prompt_user() -> AIConfig:
    """Prompt the user for input

    Returns:
        AIConfig: The AIConfig object containing the user's input
    """
    ai_name = ""
    # Load the default config
    if path.exists("autogpt/config/initial.yaml"):
        with open("autogpt/config/initial.yaml", encoding="utf-8") as file:
            config_params = load(file, Loader=FullLoader)
            ai_name = config_params["ai_name"]
            ai_role = config_params["ai_role"]
            ai_goals = config_params["ai_goals"]
            log_ai_config(config_params)
            return AIConfig(ai_name, ai_role, ai_goals)
    else:
        logger.typewriter_log("No default config found. Please enter the details below.")
    
    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
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
