"""Set up the AI and its goals"""
from colorama import Fore, Style

from autogpt import utils
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger
from autogpt.utils import clean_input


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


def get_ai_config() -> AIConfig:
    """
    Get AIConfig class object that contains the configuration information for the AI

    Returns:
        AIConfig: The AIConfig object
    """
    CFG = Config()

    ai_config = AIConfig.load(CFG.ai_settings_file)
    if CFG.skip_reprompt and ai_config.ai_name:
        logger.typewriter_log("Name :", Fore.GREEN, ai_config.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, ai_config.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{ai_config.ai_goals}")
    elif ai_config.ai_name:
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {ai_config.ai_name}?",
            speak_text=True,
        )
        should_continue = clean_input(
            f"""Continue with the last settings?
Name:  {ai_config.ai_name}
Role:  {ai_config.ai_role}
Goals: {ai_config.ai_goals}
Continue (y/n): """
        )
        if should_continue.lower() == "n":
            ai_config = AIConfig()

    if not ai_config.ai_name:
        ai_config = prompt_user()
        ai_config.save(CFG.ai_settings_file)

    return ai_config
