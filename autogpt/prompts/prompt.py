from colorama import Fore

from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.config.prompt_config import PromptConfig
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import logger
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import generate_aiconfig_manual, prompt_user
from autogpt.utils import clean_input

CFG = Config()

DEFAULT_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)


def build_default_prompt_generator() -> PromptGenerator:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Initialize the PromptConfig object and load the file set in the main config (default: prompts_settings.yaml)
    prompt_config = PromptConfig(CFG.prompt_settings_file)

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


def construct_main_ai_config(config_file: str = None) -> AIConfig:
    """Construct the prompt for the AI to respond to

    Args:
        config_file (str, optional): The path to the AI configuration file. If not provided, the default SAVE_FILE will be used.

    Returns:
        AIConfig: The chosen or created AIConfig instance
    """
    if config_file is None:
        config_file = AIConfig.SAVE_FILE

    should_save_config = True

    while True:
        logger.typewriter_log(
            f"Attempting to load configuration from: {config_file}",
            Fore.GREEN,
        )
        all_configs = AIConfig.load_all(config_file)

        if config_file in all_configs:
            config = all_configs[config_file]
            break
        elif len(all_configs) > 0:
            logger.typewriter_log(
                "Welcome back! ",
                Fore.GREEN,
                "Configuration(s) detected. Please select one:",
                speak_text=True,
            )
            for i, cfg_name in enumerate(all_configs.keys(), start=1):
                logger.typewriter_log(f"{i}. {cfg_name}")
            logger.typewriter_log(f"{len(all_configs) + 1}. Create new configuration")
            logger.typewriter_log(
                f"{len(all_configs) + 2}. Change existing configuration"
            )
            logger.typewriter_log(
                f"{len(all_configs) + 3}. Delete existing configuration"
            )
            selection = clean_input("Please choose a number: ")
            if selection.isdigit() and 0 < int(selection) <= len(all_configs) + 3:
                if int(selection) == len(all_configs) + 1:
                    # Create new config: to be defined
                    # Add the code to create a new configuration here
                    config = AIConfig()

                    # Initialize num_goals with an invalid value to enter the loop
                    num_goals = -1

                    while num_goals < 0 or num_goals > 20:
                        # Ask the user for the number of goals
                        num_goals_input = clean_input(
                            "How many goals do you want to set? (0-20): "
                        )

                        # Validate the input
                        if num_goals_input == "":
                            num_goals = 5
                            logger.typewriter_log(
                                "No input detected. Falling back to the default: 5 goals.",
                                Fore.YELLOW,
                            )
                        elif num_goals_input.isdigit():
                            num_goals = int(num_goals_input)
                            if num_goals > 20:
                                logger.typewriter_log(
                                    "More than 20 goals can be difficult to manage and should be set manually in the settings file.",
                                    Fore.YELLOW,
                                )
                            elif num_goals == 0:
                                num_goals = 5
                                logger.typewriter_log(
                                    "Falling back to the default of 5 goals.",
                                    Fore.YELLOW,
                                )
                        else:
                            logger.typewriter_log(
                                "Invalid input. Please enter a number between 0 and 20.",
                                Fore.YELLOW,
                            )

                    # Define the new configuration using user prompts
                    config = generate_aiconfig_manual(
                        max_goals=num_goals, config_file=config_file
                    )

                    # Check if required values are missing
                    if (
                        not (config.ai_name and config.ai_role and config.ai_goals)
                        or config.ai_name == "null"
                        or config.ai_name == ""
                        or config.ai_role == "null"
                        or config.ai_role == ""
                        or not config.ai_goals
                    ):
                        logger.typewriter_log(
                            "Required values are missing.",
                            Fore.RED,
                        )

                        # Prompt user if they want to save the configuration with default values
                        response = clean_input(
                            "Do you want to save the configuration with default values? (y/n): "
                        )

                        if response.lower() == "y":
                            # Assign default values if required
                            if (
                                not config.ai_name
                                or config.ai_name == "null"
                                or config.ai_name == ""
                            ):
                                config.ai_name = "Entrepreneur-GPT"
                            if (
                                not config.ai_role
                                or config.ai_role == "null"
                                or config.ai_role == ""
                            ):
                                config.ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."
                            if not config.ai_goals:
                                config.ai_goals = [
                                    "Increase net worth",
                                    "Grow Twitter Account",
                                    "Develop and manage multiple businesses autonomously",
                                ]

                            # Save the new configuration
                            config.save(config_file, append=True)
                        else:
                            should_save_config = False
                            logger.typewriter_log(
                                "New configuration not saved.", Fore.RED
                            )
                        break
                    else:
                        # Save the new configuration
                        config.save(config_file, append=True)
                    break
                elif int(selection) == len(all_configs) + 2:
                    # Edit config: Prompt user to select configuration to be edited
                    logger.typewriter_log(
                        "Please select configuration you want to change:",
                        Fore.GREEN,
                        speak_text=True,
                    )
                    while True:
                        for i, cfg_name in enumerate(all_configs.keys(), start=1):
                            logger.typewriter_log(f"{i}. {cfg_name}")
                        logger.typewriter_log(
                            f"{len(all_configs) + 1}. Go back to main menu"
                        )
                        selection = clean_input("Please choose a number: ")
                        if (
                            selection.isdigit()
                            and 0 <= int(selection) <= len(all_configs) + 1
                        ):
                            if int(selection) == len(all_configs) + 1:
                                # Go back to main menu
                                logger.typewriter_log(
                                    "Going back to main menu.",
                                    Fore.GREEN,
                                )
                                break
                            else:
                                # Edit the configuration
                                ai_name_to_edit = list(all_configs.keys())[
                                    int(selection) - 1
                                ]
                                config_to_edit = all_configs[ai_name_to_edit]

                                logger.typewriter_log(
                                    f"Editing the configuration for: {ai_name_to_edit}.",
                                    Fore.GREEN,
                                )

                                current_goal_count = len(config_to_edit.ai_goals)
                                logger.typewriter_log(
                                    f"The AI currently has {current_goal_count} goals.",
                                    Fore.GREEN,
                                )

                                num_goals_input = clean_input(
                                    "Enter a new goal count (0-20) or press Enter to keep the current count: "
                                )
                                if (
                                    num_goals_input.isdigit()
                                    and 0 <= int(num_goals_input) <= 20
                                ):
                                    num_goals = int(num_goals_input)
                                else:
                                    num_goals = current_goal_count

                                logger.typewriter_log(
                                    f"The AI will now have {num_goals} goals.",
                                    Fore.GREEN,
                                )

                                config = generate_aiconfig_manual(
                                    ai_name=ai_name_to_edit,
                                    max_goals=num_goals,
                                    config_file=config_file,
                                )

                                # Check if required values are missing
                                if (
                                    not (
                                        config.ai_name
                                        and config.ai_role
                                        and config.ai_goals
                                    )
                                    or config.ai_name == "null"
                                    or config.ai_name == ""
                                    or config.ai_role == "null"
                                    or config.ai_role == ""
                                    or not config.ai_goals
                                ):
                                    logger.typewriter_log(
                                        "Required values are missing, new configuration not saved.",
                                        Fore.RED,
                                    )
                                else:
                                    # Save the edited configuration and break the loop
                                    logger.typewriter_log(
                                        f"Saving the edited configuration for: {ai_name_to_edit}.",
                                        Fore.GREEN,
                                    )
                                    config.save(
                                        config_file,
                                        append=True,
                                        old_ai_name=ai_name_to_edit,
                                    )
                                break
                elif int(selection) == len(all_configs) + 3:
                    # Delete config
                    # Add the code to delete an existing configuration here
                    logger.typewriter_log(
                        "Please select configuration to be deleted:",
                        Fore.GREEN,
                        speak_text=True,
                    )
                    while True:  # And this loop
                        for i, cfg_name in enumerate(all_configs.keys(), start=1):
                            logger.typewriter_log(f"{i}. {cfg_name}")
                        logger.typewriter_log(
                            f"{len(all_configs) + 1}. Go back to main menu"
                        )
                        selection = clean_input("Please choose a number: ")
                        if (
                            selection.isdigit()
                            and 0 <= int(selection) <= len(all_configs) + 1
                        ):
                            if int(selection) == len(all_configs) + 1:
                                # Go back to main menu
                                logger.typewriter_log(
                                    "Going back to main menu.",
                                    Fore.GREEN,
                                )
                                break
                            else:
                                # Delete the configuration
                                ai_name_to_delete = list(all_configs.keys())[
                                    int(selection) - 1
                                ]
                                logger.typewriter_log(
                                    f"Deleting the configuration for: {ai_name_to_delete}.",
                                    Fore.RED,
                                )
                                AIConfig().delete(
                                    config_file,
                                    ai_name=ai_name_to_delete,
                                )
                                break
                else:
                    config_name = list(all_configs.keys())[int(selection) - 1]
                    config = all_configs[config_name]
                    break
            else:
                logger.typewriter_log(
                    "Invalid selection. Please enter a valid number.",
                    Fore.RED,
                )
        else:
            config = AIConfig()
            config = prompt_user()
            config.save(config_file)
            break

    if should_save_config == False:
        # User choose (n)o so call the function recursively
        config = AIConfig()
        return construct_main_ai_config(config_file)

    if CFG.skip_reprompt and config.ai_name:
        logger.typewriter_log("Name :", Fore.GREEN, config.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, config.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{config.ai_goals}")
        logger.typewriter_log(
            "API Budget:",
            Fore.GREEN,
            "infinite" if config.api_budget <= 0 else f"${config.api_budget}",
        )

    if CFG.restrict_to_workspace:
        logger.typewriter_log(
            "NOTE:All files/directories created by this agent can be found inside its workspace at:",
            Fore.YELLOW,
            f"{CFG.workspace_path}",
        )
    # set the total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(config.api_budget)

    # Agent Created, print message
    logger.typewriter_log(
        config.ai_name,
        Fore.LIGHTBLUE_EX,
        "has been created with the following details:",
        speak_text=True,
    )

    # Print the ai config details
    # Name
    logger.typewriter_log("Name:", Fore.GREEN, config.ai_name, speak_text=False)
    # Role
    logger.typewriter_log("Role:", Fore.GREEN, config.ai_role, speak_text=False)
    # Goals
    logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
    for goal in config.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)

    return config
