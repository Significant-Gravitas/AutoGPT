from typing import Optional

from colorama import Fore

from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.config.prompt_config import PromptConfig
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import logger
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

DEFAULT_TRIGGERING_PROMPT = "Determine exactly one command to use, and respond using the JSON schema specified previously:"


def build_default_prompt_generator(config: Config) -> PromptGenerator:
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
