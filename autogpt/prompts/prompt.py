from colorama import Fore

from autogpt.config.ai_config import AIConfigBroker
from autogpt.config.config import Config
from autogpt.logs import logger
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

CFG = Config()
MAX_AI_CONFIG = 5


def build_default_prompt_generator() -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
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
        ("Do Nothing", "do_nothing", {}),
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

def get_ai_config_index(number_of_config: int) -> int:
    """
    Prompt the user to select one of the existing AI configurations or start with new settings.
    Returns:
        int: The index of the selected AI configuration or -1 for new settings.
    """
    while True:
        user_input = clean_input(
            f"Type 1 to {number_of_config} to continue with the saved settings or 'n' to start with new settings: "
        )
        if user_input.lower() == "n":
            return -1
        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= number_of_config:
                return index - 1

def prompt_for_replacing_config(number_of_config: int) -> int:
    """
    Prompt the user to choose which AI configuration to replace when the maximum number of configurations is reached.
    Returns:
        int: The index of the AI configuration to be replaced.
    """
    while True:
        user_input = clean_input(
            f"There is a maximum of {number_of_config}. To create a new config, type the number of the config to replace (1 to {number_of_config}): "
        )
        if user_input.isdigit():
            index = int(user_input)
            if 1 <= index <= number_of_config:
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


def construct_main_ai_config() -> AIConfigBroker:
    """
    Load or create an AI configuration for the main AI assistant.
    Returns:
        AIConfig: The selected or created AI configuration.
    """
    ai_configs = AIConfigBroker(config_file=CFG.ai_settings_file)
    config_list = ai_configs.get_projects()
    number_of_config = len(config_list)
    config_number = -1

    if number_of_config == 0 or CFG.skip_reprompt:
        logger.typewriter_log(
            "skip_reprompt: Not supported in the current version",
            Fore.GREEN,
            config_list.ai_name,
        )

    if number_of_config == 1:
        config_number = 0
        ai_configs.set_project_number(config_number)
        config = ai_configs.get_current_project()
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {config['ai_name']}?",
            speak_text=True,
        )
        should_continue = clean_input(
            f"""Continue with the last settings?\nName:  {config['ai_name']}\nRole:  {config['ai_role']}\nGoals: {goals_to_string(config['ai_goals'])}\nContinue (y/n): """
        )
        if should_continue.lower() == "n":
            config_number -1

    elif number_of_config > 1:
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Select one of the following configurations : ",
            speak_text=True,
        )
        for i, config in enumerate(config_list):
            logger.typewriter_log(
                f"Config {i + 1} : ",
                Fore.GREEN,
                f"""Name:  {config['ai_name']}\nRole:  {config['ai_role']}\nGoals: {goals_to_string(config['ai_goals'])}: """,
            )

        config_number = get_ai_config_index(number_of_config)

    if config_number == -1:
        if number_of_config < MAX_AI_CONFIG:
            config_number = number_of_config
        else:
            config_number = prompt_for_replacing_config(number_of_config)

        config = prompt_user(config_number)
        ai_configs.save(CFG.ai_settings_file)

    else :
        ai_configs.set_project_number(new_project_number=config_number)
        config = ai_configs.get_current_project()
    
    return ai_configs
# =======
#     # Agent Created, print message
#     logger.typewriter_log(
#         config.ai_name,
#         Fore.LIGHTBLUE_EX,
#         "has been created with the following details:",
#         speak_text=True,
#     )

#     # Print the ai config details
#     # Name
#     logger.typewriter_log("Name:", Fore.GREEN, config.ai_name, speak_text=False)
#     # Role
#     logger.typewriter_log("Role:", Fore.GREEN, config.ai_role, speak_text=False)
#     # Goals
#     logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
#     for goal in config.ai_goals:
#         logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)

#     return config
# >>>>>>> master
