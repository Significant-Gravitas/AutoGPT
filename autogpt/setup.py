"""Set up the AI and its goals"""
import re
from typing import Optional

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


def generate_aiconfig_manual(
    ai_name: Optional[str] = None, max_goals: int = 5, config_file: Optional[str] = None
) -> AIConfig:
    """
    Interactively create an AI configuration by prompting the user to provide the name, role, and goals of the AI.

    If an AI name is provided, this function loads the existing configuration and allows the user to edit it.

    This function guides the user through a series of prompts to collect the necessary information to create
    an AIConfig object. The user will be asked to provide a name and role for the AI, as well as up to five
    goals. If the user does not provide a value for any of the fields, default values will be used.

    Returns:
        AIConfig: An AIConfig object containing the user-defined or default AI name, role, and goals.
    """
    # Load the existing configuration if provided
    if config_file:
        all_configs = AIConfig.load_all(config_file)
        config = all_configs.get(
            ai_name, None
        )  # Extract the AI config for the given AI name
    elif ai_name:
        config = AIConfig.load(ai_name)
    else:
        config = None  # or create a new AIConfig object with default values here

    # hold editing status
    if ai_name:
        editing = True
    else:
        editing = False

    # Manual Setup Intro
    if editing is not True:
        logger.typewriter_log(
            "Create an AI-Assistant:",
            Fore.GREEN,
            "Enter the name of your AI. Entering nothing will load the defaults.",
            speak_text=True,
        )

    if editing:
        logger.typewriter_log(
            f"Edit the AI name (current: '{ai_name}').",
            Fore.GREEN,
            "Leave empty to use the current name.",
            speak_text=True,
        )
    else:
        logger.typewriter_log(
            "Name your AI: ",
            Fore.GREEN,
            "For example, 'Entrepreneur-GPT'",
            speak_text=True,
        )

    ai_name = utils.clean_input("AI Name: ") or ai_name

    logger.typewriter_log(
        f"{ai_name} here!", Fore.LIGHTBLUE_EX, "I am at your service.", speak_text=True
    )

    if editing:
        logger.typewriter_log(
            "Describe your AI's role:",
            Fore.GREEN,
            f"Current: '{config.ai_role}', use [Enter] to keep the current role / save the input.",
            speak_text=True,
        )
    else:
        logger.typewriter_log(
            "Describe your AI's role:",
            Fore.GREEN,
            "For example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'",
            speak_text=True,
        )

    ai_role = utils.clean_input(f"{ai_name} is: ") or config.ai_role

    if ai_name and max_goals:
        # Edit Existing Goals
        default_goals = config.ai_goals if config else []
        if editing:
            logger.typewriter_log(
                f"Enter up to {max_goals} goals for your AI:",
                Fore.GREEN,
                "use [Enter] to keep the current goal / save the input.",
                speak_text=True,
            )
        else:
            logger.typewriter_log(
                f"Enter up to {max_goals} goals for your AI:",
                Fore.GREEN,
                "use [Enter] to save the input.",
                speak_text=True,
            )
        ai_goals = list(default_goals)  # start with a copy of the current goals

        for i in range(len(ai_goals)):
            logger.typewriter_log(
                f"Current Goal {i+1}: {ai_goals[i]}",
                Fore.LIGHTBLUE_EX,
                speak_text=False,
            )
            action = utils.clean_input(
                f"Do you want to [E]dit, [D]elete, or [K]eep this goal? "
            )
            if action.lower() == "e":
                ai_goal = utils.clean_input(
                    f"{Fore.LIGHTBLUE_EX}New Goal{Style.RESET_ALL} {i+1}: "
                )
                ai_goals[i] = ai_goal
            elif action.lower() == "d":
                del ai_goals[i]  # delete the goal

        # add new goals if there's still room
        if len(ai_goals) < max_goals:
            logger.typewriter_log(
                f"You can add up to {max_goals - len(ai_goals)} new goals.",
                Fore.GREEN,
                speak_text=False,
            )
            for i in range(len(ai_goals), max_goals):
                ai_goal = utils.clean_input(
                    f"{Fore.LIGHTBLUE_EX}New Goal{Style.RESET_ALL} {i+1}: "
                )
                if ai_goal == "":
                    break  # end the input process if user enters nothing
                ai_goals.append(ai_goal)  # add the new goal to the list
    else:
        # Entering new goals only, up to [x] (default=5) goals for the AI
        default_goals = config.ai_goals if config else []
        logger.typewriter_log(
            f"Enter up to {max_goals} goals for your AI: ",
            Fore.GREEN,
            "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously",
            speak_text=True,
        )
        logger.info("Use [Enter] to save the input.")
        ai_goals = []
        for i in range(max_goals):
            default_goal = default_goals[i] if i < len(default_goals) else None
            ai_goal = utils.clean_input(
                f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1} (current: '{default_goal}'): "
            )
            if ai_goal == "":
                if default_goal is not None:
                    ai_goals.append(default_goal)
                break
            ai_goals.append(ai_goal)

    # Get API Budget from User
    default_budget = config.api_budget if config else 0.0
    if editing:
        logger.typewriter_log(
            "Enter your budget for API calls: ",
            Fore.GREEN,
            f"Current: ${default_budget}. For example: $1.50, leave empty to keep current budget.",
            speak_text=True,
        )
        logger.info("Use [Enter] to save the input.")
    else:
        logger.typewriter_log(
            "Enter your budget for API calls: ",
            Fore.GREEN,
            "For example: $1.50, leave empty for unlimited budget.",
            speak_text=True,
        )
        logger.info("Use [Enter] to save the input.")
    api_budget_input = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}Budget{Style.RESET_ALL}: $"
    )
    if api_budget_input == "":
        api_budget = default_budget
    else:
        try:
            api_budget = float(api_budget_input.replace("$", ""))
        except ValueError:
            logger.typewriter_log(
                f"Invalid budget input. Using default budget (${default_budget}).",
                Fore.RED,
            )
            api_budget = default_budget

    return AIConfig(ai_name, ai_role, ai_goals, api_budget)


def generate_aiconfig_automatic(user_prompt: str) -> AIConfig:
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
    match = re.search(r"Name(?:\s*):(?:\s*)(.*)", output, re.IGNORECASE)
    ai_name = match.group(1) if match is not None else ""

    match = re.search(
        r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
        output,
        re.IGNORECASE | re.DOTALL,
    )
    ai_role = match.group(1).strip() if match is not None else ""

    ai_goals = re.findall(r"(?<=\n)-\s*(.*)", output)
    api_budget = 0.0  # TODO: parse api budget using a regular expression

    # Fallback to manual when automatic generation failed
    if not ai_name or not ai_role or not ai_goals:
        logger.typewriter_log(
            "Automatic generation failed. Falling back to manual generation...",
            Fore.RED,
        )
        return generate_aiconfig_manual()

    return AIConfig(ai_name, ai_role, ai_goals, api_budget)
