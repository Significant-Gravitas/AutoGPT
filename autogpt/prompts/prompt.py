"""
This module provides functions for configuring and setting up the AutoGPT system.

Functions:
- build_default_prompt_generator(): Generates a default prompt string for the user.
- get_ai_project_index(number_of_project: int) -> int: Prompts the user to select an existing AI configuration or start with new settings.
- prompt_for_replacing_project(number_of_project: int) -> int: Prompts the user to choose which AI configuration to replace when the maximum number of configurations is reached.
- goals_to_string(goals) -> str: Converts a list of goals into a formatted string for display.
- construct_main_project() -> ProjectsBroker: Loads or creates an AI configuration for the main AI assistant.
- construct_full_prompt(config : ProjectsBroker, prompt_generator: Optional[PromptGenerator] = None) -> str: Generates a prompt string for the user with information about the AI configuration.

Classes:
- None

Global Variables:
- CFG: An instance of the Config class containing system configuration settings.
- MAX_AI_CONFIG: The maximum number of AI configurations that can be stored.

Dependencies:
- colorama: A third-party module for adding colored text to the terminal output.
- typing: A built-in module for type hints and annotations.
- platform: A built-in module for obtaining information about the system platform.
- distro: A third-party module for obtaining information about the system distribution.
- ProjectsBroker: A class for managing the configuration settings for AutoGPT projects.
- AgentModel: A class representing the configuration settings for an AI agent.
- Project: A class representing an AutoGPT project.
- api_manager: A module for managing the GPT-3 API connection.
- Config: A class representing the system configuration settings.
- logger: A module for logging output to the terminal.
- PromptGenerator: A class for generating prompt strings for the user.
- prompt_user: A function for prompting the user for input.
- clean_input: A function for cleaning user input before processing.

"""
from colorama import Fore
from typing import  Optional
import platform
import distro

from autogpt.projects.project import AgentModel , Project
from autogpt.projects.projects_broker import ProjectsBroker
from autogpt.llm.api_manager import ApiManager
from autogpt.config.config import Config
from autogpt.llm import ApiManager
from autogpt.logs import logger
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

CFG = Config()
MAX_NB_PROJECT = 5

DEFAULT_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)


def build_default_prompt_generator() -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
    commands, resources, and performance evaluations.

    Returns:
        PromptGenerator: A PromptGenerator object containing the generated prompt.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint(
        "~4000 word limit for short term memory. Your short term memory is short, so"
        " immediately save important information to files."
    )
    prompt_generator.add_constraint(
        "If you are unsure how you previously did something or want to recall past"
        " events, thinking about similar events will help you remember."
    )
    prompt_generator.add_constraint("No user assistance")
    prompt_generator.add_constraint(
        'Exclusively use the commands listed in double quotes e.g. "command name"'
    )

    # Define the command list
    commands = [
        ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
    ]

    # Add commands to the PromptGenerator object
    for command_label, command_name, args in commands:
        prompt_generator.add_command(command_label, command_name, args)

    # Add resources to the PromptGenerator object
    prompt_generator.add_resource(
        "Internet access for searches and information gathering."
    )
    prompt_generator.add_resource("Long Term memory management.")
    prompt_generator.add_resource(
        "GPT-3.5 powered Agents for delegation of simple tasks."
    )
    prompt_generator.add_resource("File output.")

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation(
        "Continuously review and analyze your actions to ensure you are performing to"
        " the best of your abilities."
    )
    prompt_generator.add_performance_evaluation(
        "Constructively self-criticize your big-picture behavior constantly."
    )
    prompt_generator.add_performance_evaluation(
        "Reflect on past decisions and strategies to refine your approach."
    )
    prompt_generator.add_performance_evaluation(
        "Every command has a cost, so be smart and efficient. Aim to complete tasks in"
        " the least number of steps."
    )
    prompt_generator.add_performance_evaluation("Write all code to a file.")
    return prompt_generator

def get_ai_project_index(number_of_project: int) -> int:
    """
    Prompt the user to select one of the existing AI configurations or start with new settings.

    Args:
        number_of_project (int): The number of existing AI configurations.

    Returns:
        int: The index of the selected AI configuration or -1 for new settings.
    """
    while True:
        user_input = clean_input(
            f"Type 1 to {number_of_project} to continue with the saved settings or 'n' to start with new settings: "
        )
        if user_input.lower() == "n":
            return -1
        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= number_of_project:
                return index - 1

def prompt_for_replacing_project(number_of_project: int) -> int:
    """
    Prompt the user to choose which AI configuration to replace when the maximum number of configurations is reached.

    Args:
        number_of_project (int): The maximum number of AI configurations.

    Returns:
        int: The index of the AI configuration to be replaced.
    """
    while True:
        user_input = clean_input(
            f"There is a maximum of {number_of_project}. To create a new config, type the number of the config to replace (1 to {number_of_project}): "
        )
        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= number_of_project:
                return index - 1

def goals_to_string(goals) -> str:
    """
    Convert the list of goals into a formatted string for display.

    Args:
        goals (list): A list of goal strings.

    Returns:
        str: A formatted string containing the goals.
    """
    return "\n".join(goals)


def construct_main_project() -> ProjectsBroker:
    """
    Load or create an AI configuration for the main AI assistant.

    Returns:
        ProjectsBroker: The selected or created AI configuration.
    """
    configuration_broker = ProjectsBroker(config_file=CFG.ai_settings_file)
    project_list = configuration_broker.get_projects()
    number_of_project = len(project_list)
    project_number = -1

    if number_of_project == 0 or CFG.skip_reprompt:
        logger.typewriter_log(
            "skip_reprompt: Not supported in the current version",
            Fore.GREEN,
            project_list.agent_name,
        )

    if number_of_project == 1:
        project_number = 0
        configuration_broker.set_project_number(project_number)
        config = configuration_broker.get_current_project()

    elif number_of_project > 1:
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Select one of the following projects : ",
            speak_text=True,
        )
        for i, config in enumerate(project_list):
            logger.typewriter_log(
                f"Project {i + 1} : ",
                Fore.GREEN,
                f"""Name:  {config.lead_agent.agent_name}\n
                Role:  {config.lead_agent.agent_role}\n
                Goals: {goals_to_string(config.lead_agent.agent_goals)}: 
                API Budget: {"infinite" if config.api_budget <= 0 else f"${config.api_budget}"}
                Continue ({CFG.authorise_key}/{CFG.exit_key}): """,
            )

        project_number = get_ai_project_index(number_of_project)

    if project_number == -1:
        if number_of_project < MAX_NB_PROJECT:
            project_number = number_of_project
        else:
            project_number = prompt_for_replacing_project(number_of_project)

        config = prompt_user(project_number)
        # configuration_broker.save(CFG.ai_settings_file)

    else :
        configuration_broker.set_project_number(new_project_id=project_number)
        config = configuration_broker.get_current_project()
    

    # set the total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(config.api_budget)

    # Agent Created, print message
    logger.typewriter_log(
        config.project_name,
        Fore.LIGHTBLUE_EX,
        "has been created with the following details:",
        speak_text=True,
    )
    
    # Print the ai config details
    # Name
    logger.typewriter_log("Name:", Fore.GREEN, config.lead_agent.agent_name, speak_text=False)
    # Role
    logger.typewriter_log("Role:", Fore.GREEN, config.lead_agent.agent_role, speak_text=False)
    # Goals
    logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
    for goal in config.lead_agent.agent_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)

    logger.typewriter_log(
            "API Budget:",
            Fore.GREEN,
            "infinite" if config.api_budget <= 0 else f"${config.api_budget}",
        )

    return configuration_broker


def construct_full_prompt(config : ProjectsBroker
                            , prompt_generator: Optional[PromptGenerator] = None
    ) -> str:
    """
    Returns a prompt to the user with the class information in an organized fashion.

    Args:
        config (ProjectsBroker): The current AI configuration.
        prompt_generator (Optional[PromptGenerator]): An optional PromptGenerator object.

    Returns:
        str: A string containing the initial prompt for the user including the agent_name,
             agent_role, agent_goals , and api_budget.
    """

    all_projects = config.get_projects()
    if config.get_current_project_id() < 0 or config.get_current_project_id() >= len(all_projects):
        raise ValueError("No project is currently selected.")

    current_project = config.get_current_project()

    prompt_start = (
        "Your decisions must always be made independently without"
        " seeking user assistance. Play to your strengths as an LLM and pursue"
        " simple strategies with no legal complications."
        ""
    )

    from autogpt.config import Config
    from autogpt.prompts.prompt import build_default_prompt_generator

    cfg = Config()
    if prompt_generator is None:
        prompt_generator = build_default_prompt_generator()

    #prompt_generator.project_name = current_project.project_name

    prompt_generator.goals = current_project.lead_agent.agent_goals
    prompt_generator.name = current_project.lead_agent.agent_name
    prompt_generator.role = current_project.lead_agent.agent_role
    prompt_generator.command_registry = current_project.lead_agent.command_registry
    for plugin in cfg.plugins:
        if not plugin.can_handle_post_prompt():
            continue
        prompt_generator = plugin.post_prompt(prompt_generator)

    if cfg.execute_local_commands:
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
    for i, goal in enumerate(current_project.lead_agent.agent_goals):
        full_prompt += f"{i+1}. {goal}\n"
    if current_project.api_budget > 0.0:
        full_prompt += f"\nIt takes money to let you run. Your API budget is ${current_project.api_budget:.3f}"
    
    current_project.lead_agent.prompt_generator = prompt_generator
    full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
    return full_prompt