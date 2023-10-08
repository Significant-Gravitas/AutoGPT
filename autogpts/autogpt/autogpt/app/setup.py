"""Set up the AI and its goals"""
import logging
import re
from typing import Optional

from colorama import Fore, Style
from jinja2 import Template

from autogpt.app import utils
from autogpt.config import Config
from autogpt.config.ai_profile import AIProfile
from autogpt.core.resource.model_providers import ChatMessage, ChatModelProvider
from autogpt.logs.helpers import user_friendly_output
from autogpt.prompts.default_prompts import (
    DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_USER_DESIRE_PROMPT,
)

logger = logging.getLogger(__name__)


async def interactive_ai_profile_setup(
    config: Config,
    llm_provider: ChatModelProvider,
    ai_profile_template: Optional[AIProfile] = None,
) -> AIProfile:
    """Prompt the user for input

    Params:
        config (Config): The Config object
        ai_profile_template (AIProfile): The AIProfile object to use as a template

    Returns:
        AIProfile: The AIProfile object tailored to the user's input
    """

    # Construct the prompt
    user_friendly_output(
        title="Welcome to AutoGPT! ",
        message="run with '--help' for more information.",
        title_color=Fore.GREEN,
    )

    ai_profile_template_provided = ai_profile_template is not None and any(
        [
            ai_profile_template.ai_goals,
            ai_profile_template.ai_name,
            ai_profile_template.ai_role,
        ]
    )

    user_desire = ""
    if not ai_profile_template_provided:
        # Get user desire if command line overrides have not been passed in
        user_friendly_output(
            title="Create an AI-Assistant:",
            message="input '--manual' to enter manual mode.",
            title_color=Fore.GREEN,
        )

        user_desire = await utils.clean_input(
            config, f"{Fore.LIGHTBLUE_EX}I want AutoGPT to{Style.RESET_ALL}: "
        )

    if user_desire.strip() == "":
        user_desire = DEFAULT_USER_DESIRE_PROMPT  # Default prompt

    # If user desire contains "--manual" or we have overridden any of the AI configuration
    if "--manual" in user_desire or ai_profile_template_provided:
        user_friendly_output(
            "",
            title="Manual Mode Selected",
            title_color=Fore.GREEN,
        )
        return await generate_aiconfig_manual(config, ai_profile_template)

    else:
        try:
            return await generate_aiconfig_automatic(user_desire, config, llm_provider)
        except Exception as e:
            user_friendly_output(
                title="Unable to automatically generate AI Config based on user desire.",
                message="Falling back to manual mode.",
                title_color=Fore.RED,
            )
            logger.debug(f"Error during AIProfile generation: {e}")

            return await generate_aiconfig_manual(config)


async def generate_aiconfig_manual(
    config: Config, ai_profile_template: Optional[AIProfile] = None
) -> AIProfile:
    """
    Interactively create an AI configuration by prompting the user to provide the name, role, and goals of the AI.

    This function guides the user through a series of prompts to collect the necessary information to create
    an AIProfile object. The user will be asked to provide a name and role for the AI, as well as up to five
    goals. If the user does not provide a value for any of the fields, default values will be used.

    Params:
        config (Config): The Config object
        ai_profile_template (AIProfile): The AIProfile object to use as a template

    Returns:
        AIProfile: An AIProfile object containing the user-defined or default AI name, role, and goals.
    """

    # Manual Setup Intro
    user_friendly_output(
        title="Create an AI-Assistant:",
        message="Enter the name of your AI and its role below. Entering nothing will load"
        " defaults.",
        title_color=Fore.GREEN,
    )

    if ai_profile_template and ai_profile_template.ai_name:
        ai_name = ai_profile_template.ai_name
    else:
        ai_name = ""
        # Get AI Name from User
        user_friendly_output(
            title="Name your AI:",
            message="For example, 'Entrepreneur-GPT'",
            title_color=Fore.GREEN,
        )
        ai_name = await utils.clean_input(config, "AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    user_friendly_output(
        title=f"{ai_name} here!",
        message="I am at your service.",
        title_color=Fore.LIGHTBLUE_EX,
    )

    if ai_profile_template and ai_profile_template.ai_role:
        ai_role = ai_profile_template.ai_role
    else:
        # Get AI Role from User
        user_friendly_output(
            title="Describe your AI's role:",
            message="For example, 'an AI designed to autonomously develop and run businesses with"
            " the sole goal of increasing your net worth.'",
            title_color=Fore.GREEN,
        )
        ai_role = await utils.clean_input(config, f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the"
        " sole goal of increasing your net worth."

    if ai_profile_template and ai_profile_template.ai_goals:
        ai_goals = ai_profile_template.ai_goals
    else:
        # Enter up to 5 goals for the AI
        user_friendly_output(
            title="Enter up to 5 goals for your AI:",
            message="For example: \nIncrease net worth, Grow Twitter Account, Develop and manage"
            " multiple businesses autonomously'",
            title_color=Fore.GREEN,
        )
        logger.info("Enter nothing to load defaults, enter nothing when finished.")
        ai_goals = []
        for i in range(5):
            ai_goal = await utils.clean_input(
                config, f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: "
            )
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
    user_friendly_output(
        title="Enter your budget for API calls:",
        message="For example: $1.50",
        title_color=Fore.GREEN,
    )
    logger.info("Enter nothing to let the AI run without monetary limit")
    api_budget_input = await utils.clean_input(
        config, f"{Fore.LIGHTBLUE_EX}Budget{Style.RESET_ALL}: $"
    )
    if api_budget_input == "":
        api_budget = 0.0
    else:
        try:
            api_budget = float(api_budget_input.replace("$", ""))
        except ValueError:
            user_friendly_output(
                level=logging.WARNING,
                title="Invalid budget input.",
                message="Setting budget to unlimited.",
                title_color=Fore.RED,
            )
            api_budget = 0.0

    return AIProfile(
        ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals, api_budget=api_budget
    )


async def generate_aiconfig_automatic(
    user_prompt: str,
    config: Config,
    llm_provider: ChatModelProvider,
) -> AIProfile:
    """Generates an AIProfile object from the given string.

    Returns:
    AIProfile: The AIProfile object tailored to the user's input
    """

    system_prompt = DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC
    prompt_ai_profile_automatic = Template(
        DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC
    ).render(user_prompt=user_prompt)
    # Call LLM with the string as user input
    output = (
        await llm_provider.create_chat_completion(
            [
                ChatMessage.system(system_prompt),
                ChatMessage.user(prompt_ai_profile_automatic),
            ],
            config.smart_llm,
        )
    ).response["content"]

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

    return AIProfile(
        ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals, api_budget=api_budget
    )
