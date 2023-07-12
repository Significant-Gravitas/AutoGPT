import platform
from typing import Optional

import distro
from colorama import Fore

from autogpt.app.setup import prompt_user
from autogpt.config import AIConfig, Config, PromptConfig
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import logger
from autogpt.models.command_registry import CommandRegistry
from autogpt.prompts.generator import PromptGenerator
from autogpt.utils import clean_input


def build_default_prompt_generator(
    config: Config, ai_config: AIConfig, command_registry: CommandRegistry
) -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Initialize the PromptConfig object and load the file set in the main config (default: prompts_settings.yaml)
    prompt_config = PromptConfig(config.prompt_settings_file)

    # Add constraints to the PromptGenerator object
    for constraint in prompt_config.constraints:
        prompt_generator.add_constraint(constraint)

    # Add resources to the PromptGenerator object
    for resource in prompt_config.resources:
        prompt_generator.add_resource(resource)

    # Add performance evaluations to the PromptGenerator object
    for performance_evaluation in prompt_config.performance_evaluations:
        prompt_generator.add_performance_evaluation(performance_evaluation)

    prompt_generator.goals = ai_config.ai_goals
    prompt_generator.name = ai_config.ai_name
    prompt_generator.role = ai_config.ai_role
    prompt_generator.command_registry = command_registry

    return prompt_generator


def construct_main_ai_config(
    config: Config,
    name: Optional[str] = None,
    role: Optional[str] = None,
    goals: tuple[str] = tuple(),
) -> AIConfig:
    """Construct the prompt for the AI to respond to

    Returns:
        str: The prompt string
    """
    ai_config = AIConfig.load(config.ai_settings_file)

    # Apply overrides
    if name:
        ai_config.ai_name = name
    if role:
        ai_config.ai_role = role
    if goals:
        ai_config.ai_goals = list(goals)

    if (
        all([name, role, goals])
        or config.skip_reprompt
        and all([ai_config.ai_name, ai_config.ai_role, ai_config.ai_goals])
    ):
        logger.typewriter_log("Name :", Fore.GREEN, ai_config.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, ai_config.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{ai_config.ai_goals}")
        logger.typewriter_log(
            "API Budget:",
            Fore.GREEN,
            "infinite" if ai_config.api_budget <= 0 else f"${ai_config.api_budget}",
        )
    elif all([ai_config.ai_name, ai_config.ai_role, ai_config.ai_goals]):
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {ai_config.ai_name}?",
            speak_text=True,
        )
        should_continue = clean_input(
            config,
            f"""Continue with the last settings?
Name:  {ai_config.ai_name}
Role:  {ai_config.ai_role}
Goals: {ai_config.ai_goals}
API Budget: {"infinite" if ai_config.api_budget <= 0 else f"${ai_config.api_budget}"}
Continue ({config.authorise_key}/{config.exit_key}): """,
        )
        if should_continue.lower() == config.exit_key:
            ai_config = AIConfig()

    if any([not ai_config.ai_name, not ai_config.ai_role, not ai_config.ai_goals]):
        ai_config = prompt_user(config)
        ai_config.save(config.ai_settings_file)

    if config.restrict_to_workspace:
        logger.typewriter_log(
            "NOTE:All files/directories created by this agent can be found inside its workspace at:",
            Fore.YELLOW,
            f"{config.workspace_path}",
        )
    # set the total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(ai_config.api_budget)

    # Agent Created, print message
    logger.typewriter_log(
        ai_config.ai_name,
        Fore.LIGHTBLUE_EX,
        "has been created with the following details:",
        speak_text=True,
    )

    # Print the ai_config details
    # Name
    logger.typewriter_log("Name:", Fore.GREEN, ai_config.ai_name, speak_text=False)
    # Role
    logger.typewriter_log("Role:", Fore.GREEN, ai_config.ai_role, speak_text=False)
    # Goals
    logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
    for goal in ai_config.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)

    return ai_config


def construct_full_prompt(
    config: Config,
    ai_config: AIConfig,
    command_registry: CommandRegistry,
    prompt_generator: Optional[PromptGenerator] = None,
) -> str:
    """
    Returns a prompt to the user with the class information in an organized fashion.

    Parameters:
        config (Config): The main config object.
        ai_config (AIConfig): The AI config object.
        prompt_generator (PromptGenerator): The prompt generator object.

    Returns:
        full_prompt (str): A string containing the initial prompt for the user
          including the ai_name, ai_role, ai_goals, and api_budget.
    """

    prompt_start = (
        "Your decisions must always be made independently without"
        " seeking user assistance. Play to your strengths as an LLM and pursue"
        " simple strategies with no legal complications."
        ""
    )

    if prompt_generator is None:
        prompt_generator = build_default_prompt_generator(
            config, ai_config, command_registry
        )

    for plugin in config.plugins:
        if not plugin.can_handle_post_prompt():
            continue
        prompt_generator = plugin.post_prompt(prompt_generator)

    if config.execute_local_commands:
        # add OS info to prompt
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )

        prompt_start += f"\nThe OS you are running on is: {os_info}"

    # Construct full prompt
    full_prompt = f"You are {prompt_generator.name}, {prompt_generator.role}\n{prompt_start}\n\nGOALS:\n\n"
    for i, goal in enumerate(ai_config.ai_goals):
        full_prompt += f"{i+1}. {goal}\n"
    if ai_config.api_budget > 0.0:
        full_prompt += f"\nIt takes money to let you run. Your API budget is ${ai_config.api_budget:.3f}"
    ai_config.prompt_generator = prompt_generator
    full_prompt += f"\n\n{prompt_generator.generate_prompt_string(config)}"
    return full_prompt
