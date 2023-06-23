import os
import re
from typing import Any, Dict, List, Optional, Tuple

from colorama import Fore, Style

from autogpt import utils
from autogpt.commands.file_operations import handle_ingest
from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.config.prompt_config import PromptConfig
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import logger
from autogpt.prompts.default_prompts import DEFAULT_USER_DESIRE_PROMPT
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import generate_aiconfig_automatic
from autogpt.utils import clean_input

DEFAULT_TRIGGERING_PROMPT = "Determine exactly one command to use, and respond using the JSON schema specified previously:"


# Helper functions
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


def get_valid_input(config: Config, num_options: int) -> int:
    """
    Check if user input number is within valid range.

    Params:
        num_options: The total number of options available.

    Returns:
        The selected option as an integer.

    """
    while True:
        try:
            selection = int(clean_input(config, "Choose a number: "))
            if (
                1 <= selection <= num_options
            ):  # The number must be within the valid range
                return selection
        except ValueError:
            logger.typewriter_log("Invalid input. Please enter a number.", Fore.RED)


def select_config_action(
    config: Config, all_configs: Dict[str, Any], task: str
) -> Optional[Dict[str, Any]]:
    """
    Prompt user for the selection of an AI configuration or option.

    Params:
        all_configs (Dict[str, Any]): A dictionary of all available AI configurations.
        task (str): The task being performed, either "create" or "edit".

    Returns:
        Optional[Dict[str, Any]]: The selected AI configuration, or user goes back to the main menu.
    """
    hold_to_ingest = {}

    if task in ["edit", "view", "delete"]:
        cfg_count = len(all_configs)
        logger.typewriter_log(
            f"Select a configuration to {task}:", Fore.GREEN, speak_text=True
        )
        for i, cfg_name in enumerate(all_configs.keys(), start=1):
            logger.typewriter_log(f"{i}. {cfg_name}")
    else:
        cfg_count = 0
        logger.typewriter_log(
            f"Choose a directory or file to {task}:", Fore.GREEN, speak_text=True
        )

        items = os.listdir(config.workspace_path)
        directories = [
            item
            for item in items
            if os.path.isdir(os.path.join(config.workspace_path, item))
        ]
        files = [
            item
            for item in items
            if os.path.isfile(os.path.join(config.workspace_path, item))
        ]

        for i, directory in enumerate(directories, start=1):
            logger.typewriter_log(f"{i}. [ dir] --- ./{directory}")
            hold_to_ingest[i] = os.path.join(config.workspace_path, directory)
            cfg_count += 1

        for i, file in enumerate(files, start=len(directories) + 1):
            logger.typewriter_log(f"{i}. [file] --- ./{file}")
            hold_to_ingest[i] = os.path.join(config.workspace_path, file)
            cfg_count += 1

    logger.typewriter_log(f"{cfg_count+1}. Go back to main menu")

    selection = get_valid_input(config, cfg_count + 1)
    if selection == cfg_count + 1:
        return None
    else:
        if task == "ingest":
            return hold_to_ingest.get(selection)
        else:
            return all_configs[list(all_configs.keys())[selection - 1]]


def prompt_for_num_goals(
    config: Config, task: str, current_goal_count: Optional[int] = None
) -> int:
    """
    Prompt the user to enter the number of goals for the AI.

    Parameters:
        task (str): The task being performed, either "create" or "edit".
        current_goal_count (Optional[int]): The current number of goals (optional).

    Returns:
        int: The number of goals entered by the user.

    """
    num_goals: int = current_goal_count if current_goal_count is not None else -1

    prompts = {
        "create": "Set number of goals (0-20): ",
        "edit": f"AI has {current_goal_count} goals. Change? (0-20): ",
    }[task]

    fallback_msg = {
        "create": "No input, defaulting to 5 goals.",
        "edit": f"No change, {current_goal_count or 0} goals.",
    }[task]

    while True:
        logger.typewriter_log(
            "Decide how many goals your AI has to fulfill:",
            Fore.GREEN,
            speak_text=True,
        )

        num_goals_input = clean_input(config, prompts)

        if num_goals_input == "":
            if task == "create":
                num_goals = 5
            elif task == "edit" and current_goal_count is not None:
                num_goals = current_goal_count
            logger.typewriter_log(fallback_msg, Fore.LIGHTBLUE_EX)
            break

        elif num_goals_input.isdigit():
            num_goals = int(num_goals_input)
            if num_goals > 20:
                logger.typewriter_log(
                    "Over 20 goals? Please set manually in settings file.", Fore.RED
                )
                continue
            elif num_goals == 0 and task == "create":
                logger.typewriter_log(
                    "No goals set, using default of 5.", Fore.LIGHTBLUE_EX
                )
                num_goals = 5
            break
        else:
            logger.typewriter_log("Invalid input, enter a number (0-20).", Fore.RED)

    return num_goals


def check_ai_name_exists(config: Config, name: str) -> bool:
    """Check if a name exists in the current configurations."""
    all_configs, _ = AIConfig.load_all(
        config.ai_settings_filepath
    )  # , _ to ignore returned message from load_all
    for tmp_cfg in all_configs.values():
        if tmp_cfg.ai_name == name:
            return True
    return False


def generate_unique_name(config: Config, base: str) -> str:
    """Generate unique AI name."""
    i: int = 1
    while check_ai_name_exists(config, f"{base}-{i}"):
        i += 1
    return f"{base}-{i}"


def validate_input(prompt_text: str) -> str:
    """Validate user input."""
    while True:
        user_input = input(prompt_text).strip()
        # Validate user input against pattern
        if re.match(r"^[a-zA-Z0-9-]*$", user_input) and not user_input.startswith("-"):
            return user_input
        print(
            "Input can only contain alphanumeric characters and dashes, and cannot start with '-'. Please try again."
        )


# Start AI prompts
def welcome_prompt(config: Config) -> Optional[AIConfig]:
    """
    Display the welcome message and prompt the user for their desire to create an AI Assistant.

    Returns:
        Optional[AIConfig]: If manual mode is selected, returns the result of the `handle_config` function with None as the first argument.
        If automatic mode is selected, returns the result of the `generate_aiconfig_automatic` function.
        If an exception occurs during automatic mode, falls back to manual mode and returns the result of the `handle_config` function.
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

    user_desire = utils.clean_input(config, f"I want Auto-GPT to: ")

    if user_desire == "":
        user_desire = DEFAULT_USER_DESIRE_PROMPT  # Default prompt

    # If user desire contains "--manual"
    if "--manual" in user_desire:
        logger.typewriter_log(
            "Manual Mode Selected.",
            Fore.LIGHTBLUE_EX,
            speak_text=True,
        )
        return handle_config(config, None, "create")

    else:
        try:
            return generate_aiconfig_automatic(config, user_desire)
        except Exception as e:
            logger.typewriter_log(
                "Unable to automatically generate AI Config based on user desire.",
                Fore.RED,
                "Falling back to manual mode.",
                speak_text=True,
            )

            return handle_config(config, None, "create")


def start_prompt(config: Config, tmp_cfg: AIConfig, sad: Optional[Any] = None) -> None:
    """
    Display the start prompt for Auto-GPT with the provided configuration.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration.
        sad (Optional[Any]): An optional parameter. [Add description of the parameter here]

    Returns:
        None

    """
    from autogpt.main import start_agent_directly

    if config.skip_reprompt and tmp_cfg.ai_name:
        logger.typewriter_log("Name :", Fore.GREEN, tmp_cfg.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, tmp_cfg.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{tmp_cfg.ai_goals}")
        logger.typewriter_log(
            "Budget:",
            Fore.GREEN,
            "infinite" if tmp_cfg.api_budget <= 0 else f"${str(tmp_cfg.api_budget)}",
        )

    if config.restrict_to_workspace:
        logger.typewriter_log(
            "NOTE:All files/directories created by this agent can be found inside its workspace at:",
            Fore.YELLOW,
        )
        logger.typewriter_log(f"- {config.workspace_path}", Fore.YELLOW)
    # Set total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(tmp_cfg.api_budget)

    # Agent Created
    logger.typewriter_log(
        "Auto-GPT has started with the following details:",
        Fore.LIGHTBLUE_EX,
        speak_text=True,
    )

    logger.typewriter_log("Name:", Fore.GREEN, tmp_cfg.ai_name, speak_text=False)
    logger.typewriter_log("Role:", Fore.GREEN, tmp_cfg.ai_role, speak_text=False)
    logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
    for goal in tmp_cfg.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)
    logger.typewriter_log(
        "Budget:", Fore.GREEN, f"${str(tmp_cfg.api_budget)}", speak_text=False
    )
    if hasattr(tmp_cfg, "plugins"):
        logger.typewriter_log("Plugins:", Fore.GREEN, "", speak_text=False)
        for plugin in tmp_cfg.plugins:
            logger.typewriter_log("-", Fore.GREEN, plugin, speak_text=False)

    # don't exec when called from main
    if sad:
        start_agent_directly(tmp_cfg, config)


# Main menu functions
def display_main_menu(
    all_configs: Dict[str, Any], menu_options: List[Tuple[str, Any]]
) -> None:
    """
    Display the main menu for Auto-GPT.

    Parameters:
        all_configs (Dict[str, Any]): A dictionary containing all configurations.
        menu_options (List[Tuple[str, Any]]): A list of menu options, where each option is represented as a tuple
            containing the option text and its corresponding value.

    Returns:
        None

    """
    cfg_count = len(all_configs)

    logger.typewriter_log(
        "Select configuration or option:", Fore.GREEN, speak_text=True
    )
    for i, cfg_name in enumerate(all_configs.keys(), start=1):
        logger.typewriter_log(f"{i}. {cfg_name}")
    for i, (option_text, _) in enumerate(menu_options, start=cfg_count + 1):
        logger.typewriter_log(f"{i}. {option_text}")


def main_menu(config: Config) -> Optional[AIConfig]:
    """
    Display the main menu and handle user interactions.

    Returns:
        Optional[AIConfig]: The selected or created AI configuration, or None if no configuration was selected or created.
    """
    tmp_cfg = None
    config_file = config.ai_settings_filepath

    menu_options = [
        (
            "Create new configuration",
            lambda tmp_cfg: handle_config(config, tmp_cfg, "create"),
        ),
        (
            "Edit existing configuration",
            lambda tmp_cfg: handle_config(config, tmp_cfg, "edit"),
        ),
        ("View existing configuration", lambda tmp_cfg: view_config(config, tmp_cfg)),
        (
            "Delete existing configuration",
            lambda tmp_cfg: delete_config(config, tmp_cfg),
        ),
        (
            "Ingest data from directory or file",
            lambda tmp_cfg, to_ingest: handle_ingest(config, to_ingest),
        ),
        # add more options here...
    ]

    while True:
        all_configs, message = AIConfig.load_all(config_file)
        all_configs_keys = list(all_configs.keys())
        cfg_count = len(all_configs)  # Configurations count
        total_options = cfg_count + len(menu_options)

        logger.typewriter_log(
            f"Status of {config.ai_settings_file}:", Fore.GREEN, message
        )

        if all_configs:  # Only log if all_configs is not empty
            logger.typewriter_log(
                f"Attempting to load {config.ai_settings_file}:",
                Fore.GREEN,
                "load successful.",
            )

        if cfg_count > 0:
            logger.typewriter_log(
                "Main menu:",
                Fore.GREEN,
                f"{cfg_count} {'configuration' if cfg_count == 1 else 'configurations'} detected.",
                speak_text=True,
            )

            display_main_menu(all_configs, menu_options)  # Displaying menu here
            selection = get_valid_input(config, total_options)

            if 1 <= selection <= cfg_count:  # If selection is a valid index
                config_name = all_configs_keys[selection - 1]
                tmp_cfg = all_configs[config_name]
                return tmp_cfg
            else:  # "Create", "Edit", "View", "Delete"
                task_index = selection - cfg_count - 1
                task = menu_options[task_index][0].split()[0].lower()
                _, action = menu_options[task_index]
                if task in ["edit", "view", "delete"]:
                    chosen_config = select_config_action(config, all_configs, task)
                    if chosen_config == None:
                        continue  # Start over
                    tmp_cfg = action(chosen_config)
                elif task in ["ingest"]:
                    chosen_config = select_config_action(config, all_configs, task)
                    if chosen_config == None:
                        continue  # Start over
                    tmp_cfg = action(config, to_ingest=chosen_config)
                else:  # On "create"
                    tmp_cfg = AIConfig()
                    tmp_cfg = action(tmp_cfg)
                if tmp_cfg:  # If an action returns a tmp_cfg
                    return tmp_cfg

        else:  # If no configurations exist
            tmp_cfg = AIConfig()
            tmp_cfg = welcome_prompt(config)
            return tmp_cfg


# Create, edit configurations
def manage_ai_name(config: Config, tmp_cfg: AIConfig, task: str) -> str:
    """
    Manage the AI name based on the task.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration.
        task (str): The task being performed, either "create" or "edit".

    Returns:
        str: The updated AI name.

    """
    prompts = {
        "create": ("Create an AI-Assistant:", "For example, 'Entrepreneur-GPT'"),
        "edit": (
            "Edit the AI name:",
            f"current name is '{tmp_cfg.ai_name if tmp_cfg else None}'. Leave empty to keep the current name.",
        ),
    }
    # Display prompt
    logger.typewriter_log(
        prompts[task][0],
        Fore.GREEN,
        prompts[task][1],
        speak_text=True,
    )

    prompt_text = "AI Name: "

    # Get name: user input
    input_name = utils.clean_input(config, prompt_text)
    ai_name = input_name if input_name or task == "create" else tmp_cfg.ai_name

    # Check if name already exists, prompt until unique name provided
    while check_ai_name_exists(config, ai_name) and (input_name or task == "create"):
        logger.typewriter_log(
            "This AI name already exists. Please choose a different name.",
            Fore.RED,
            speak_text=True,
        )
        input_name = utils.clean_input(config, validate_input(prompt_text))
    logger.typewriter_log("AI name assigned.", Fore.LIGHTBLUE_EX, speak_text=True)
    tmp_cfg.ai_name = ai_name
    return ai_name


def manage_ai_role(config: Config, tmp_cfg: AIConfig, task: str) -> str:
    """
    Manage the AI role based on the task.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration.
        task (str): The task being performed, either "create" or "edit".

    Returns:
        str: The updated AI role.

    """
    default_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."
    prompts = {
        "create": ("Describe your AI's role:", f"For example, '{default_role}'"),
        "edit": (
            "Describe your AI's role:",
            f"Current role is: '{tmp_cfg.ai_role if tmp_cfg else None}'. Leave empty to keep the current role.",
        ),
    }

    # Display prompt
    logger.typewriter_log(
        prompts[task][0],
        Fore.GREEN,
        prompts[task][1],
        speak_text=True,
    )

    # Get role: user input
    ai_role = (
        utils.clean_input(config, f"{tmp_cfg.ai_name if tmp_cfg else None} is: ")
        or tmp_cfg.ai_role
        if tmp_cfg
        else default_role
    )

    # fallback message
    if ai_role == default_role:
        logger.typewriter_log(
            f"No input, defaulting to AI role: '{default_role}'",
            Fore.LIGHTBLUE_EX,
            speak_text=True,
        )

    logger.typewriter_log("AI role assigned.", Fore.LIGHTBLUE_EX, speak_text=True)

    tmp_cfg.ai_role = ai_role
    return ai_role


def manage_ai_goals(config: Config, tmp_cfg: AIConfig, task: str) -> List[str]:
    """
    Manage the AI goals based on the task.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration.
        task (str): The task being performed, either "create" or "edit".

    Returns:
        List[str]: The updated AI goals.

    """
    # Store current_goals and max_goals
    current_goals = tmp_cfg.ai_goals if tmp_cfg else []
    max_goals = prompt_for_num_goals(config, task, len(current_goals))

    ai_goals = []
    num_default_goals = len(current_goals)

    if task == "create":
        # Create goals: new AI
        logger.typewriter_log(
            f"Enter up to {max_goals} goals for your AI: ", Fore.GREEN, speak_text=True
        )

        for i in range(max_goals):
            ai_goal = utils.clean_input(config, f"Goal {i+1}: ")
            logger.info("Use [Enter] to save the input.")
            if ai_goal == "":
                break
            ai_goals.append(ai_goal)

    else:
        # Edit or delete goals: existing AI
        logger.typewriter_log(
            f"Enter up to {max_goals} goals for your AI:", Fore.GREEN, speak_text=True
        )
        kept_goals = []

        for i in range(min(num_default_goals, max_goals)):
            logger.typewriter_log(
                f"Current Goal {i+1}: ", Fore.GREEN, f"'{current_goals[i]}'"
            )
            action = utils.clean_input(
                config, "Do you want to [E]dit, [D]elete, or [K]eep this goal? "
            )
            if action.lower() == "e":
                ai_goal = utils.clean_input(
                    config, f"{Fore.GREEN}New Goal {i+1}:{Style.RESET_ALL} "
                )
                if ai_goal != "":
                    kept_goals.append(ai_goal)
                else:
                    kept_goals.append(current_goals[i])
            elif action.lower() == "d":
                continue
            else:
                kept_goals.append(current_goals[i])

        ai_goals = kept_goals
        num_ai_goals = len(ai_goals)

        if num_ai_goals < max_goals:
            logger.typewriter_log(
                f"You can add up to {max_goals - num_ai_goals} new goals.", Fore.GREEN
            )
            # Add new goals if max_goals not reached
            for i in range(num_ai_goals, max_goals):
                ai_goal = utils.clean_input(
                    config, f"{Fore.GREEN}New Goal {i+1}:{Style.RESET_ALL} "
                )
                if ai_goal == "":
                    break
                ai_goals.append(ai_goal)
    logger.typewriter_log("AI goals assigned.", Fore.LIGHTBLUE_EX, speak_text=True)
    tmp_cfg.ai_goals = ai_goals
    return ai_goals


def manage_plugins(config: Config, tmp_cfg: AIConfig, task: str) -> List[str]:
    """
    Manage the plugins based on the task.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration.
        task (str): The task being performed, either "create" or "edit".

    Returns:
        List[str]: The updated plugins.

    """
    default_plugins = (
        config.plugins_allowlist if config and config.plugins_allowlist else []
    )
    plugins = []

    if default_plugins:
        logger.typewriter_log(
            "Add plugins from the plugins allowlist?", Fore.GREEN, speak_text=True
        )

        if task == "create":
            # Select new plugins from allowlist
            env_plugins = list(default_plugins)
            remaining_plugins = []
            for i in range(len(env_plugins)):
                action = utils.clean_input(
                    config,
                    f"{Fore.GREEN}Plugin {i+1}: {Fore.YELLOW}'{env_plugins[i]}'\n{Fore.WHITE}Do you want to [I]gnore or [A]dd this plugin?{Style.RESET_ALL} ",
                )
                if action.lower() == "a":
                    remaining_plugins.append(env_plugins[i])

            plugins = remaining_plugins
        else:
            # Edit existing plugins
            plugins = tmp_cfg.plugins
            deleted_plugins = []
            i = 0
            while i < len(plugins):
                logger.typewriter_log(
                    f"Current plugin {i+1}: ",
                    Fore.GREEN,
                    f"'{plugins[i]}'",
                    speak_text=False,
                )
                action = utils.clean_input(
                    config, f"Do you want to [D]elete or [K]eep this plugin?"
                )
                if action.lower() == "d":
                    deleted_plugins.append(plugins[i])
                    del plugins[i]
                else:
                    i += 1

            # Offer plugins from allowlist
            env_plugins = default_plugins
            for plugin in env_plugins:
                if plugin not in plugins and plugin not in deleted_plugins:
                    logger.typewriter_log(
                        f"New plugin available: ",
                        Fore.GREEN,
                        f"'{plugin}'",
                        speak_text=False,
                    )
                    action = utils.clean_input(
                        config, "Do you want to ignore [E]nter or [A]dd this plugin?"
                    )
                    if action.lower() == "a":
                        plugins.append(plugin)
    logger.typewriter_log("Plugins assigned.", Fore.LIGHTBLUE_EX, speak_text=True)
    tmp_cfg.plugins = plugins
    return plugins


def manage_api_budget(config: Config, tmp_cfg: AIConfig, task: str) -> float:
    """
    Manage the API budget based on the task.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration.
        task (str): The task being performed, either "create" or "edit".

    Returns:
        float: The updated API budget.

    """
    default_budget = tmp_cfg.api_budget if tmp_cfg else 0.0

    prompts = {
        "create": (
            "Enter your budget for API calls: ",
            "For example: $1.50, leave empty for unlimited budget.",
        ),
        "edit": (
            "Enter your budget for API calls:",
            f"Current budget is: '${default_budget}'. For example: $1.50 or leave empty to keep current budget.",
        ),
    }

    # Display prompt
    logger.typewriter_log(
        f"{prompts[task][0]}",
        Fore.GREEN,
        f"\n{prompts[task][1]}",
        speak_text=True,
    )

    # Get budget: user input
    api_budget_input = utils.clean_input(
        config, f"{Fore.WHITE}Budget: ${Style.RESET_ALL}"
    )
    api_budget = default_budget

    # Attempt to convert to float
    if api_budget_input:
        try:
            api_budget = float(api_budget_input.replace("$", ""))
        except ValueError:
            logger.typewriter_log(
                f"Invalid budget input. Using default budget (${default_budget}).",
                Fore.RED,
            )
            api_budget = default_budget

    tmp_cfg.api_budget = api_budget
    return api_budget


# View, delete configurations
def view_config(config: Config, tmp_cfg: AIConfig) -> None:
    """
    View the details of the given AI configuration.

    Parameters:
        tmp_cfg (AIConfig): The AI configuration to view.

    Returns:
        None

    """
    logger.typewriter_log(
        f"Configuration {tmp_cfg.ai_name} has the following details:",
        Fore.LIGHTBLUE_EX,
        speak_text=True,
    )
    logger.typewriter_log("Name:", Fore.GREEN, tmp_cfg.ai_name, speak_text=False)
    logger.typewriter_log("Role:", Fore.GREEN, tmp_cfg.ai_role, speak_text=False)
    logger.typewriter_log("Goals:", Fore.GREEN, speak_text=False)
    for goal in tmp_cfg.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)
    logger.typewriter_log(
        "Budget:", Fore.GREEN, f"${str(tmp_cfg.api_budget)}", speak_text=False
    )
    if hasattr(tmp_cfg, "plugins"):
        logger.typewriter_log("Plugins:", Fore.GREEN, speak_text=False)
        for plugin in tmp_cfg.plugins:
            logger.typewriter_log("-", Fore.GREEN, plugin, speak_text=False)

    # Prompt
    choice = input("Do you want to [E]dit or [R]eturn to the main menu? ").lower()

    if choice == "e":
        handle_config(config, tmp_cfg, "edit")

    logger.typewriter_log("Returning to the main menu.", speak_text=True)
    return None


def delete_config(config: Config, selected_config: AIConfig) -> None:
    """
    Delete the specified AI configuration.

    Parameters:
        selected_config (AIConfig): The AI configuration to delete.

    Returns:
        None
    """
    # Delete configuration
    try:
        logger.typewriter_log(
            f"Deleting the configuration for: {selected_config.ai_name}.",
            Fore.RED,
        )
        AIConfig().delete(
            config.ai_settings_filepath,
            selected_config.ai_name,
        )
    except ValueError as e:
        logger.typewriter_log(f"An error occurred: {e}", Fore.RED)
    else:
        logger.typewriter_log(
            f"Configuration for {selected_config.ai_name} deleted.",
            Fore.GREEN,
            speak_text=True,
        )

    logger.typewriter_log("Returning to the main menu.", speak_text=True)
    return None


# Menu handling
def handle_configs(config: Config, tmp_cfg: AIConfig, task: str) -> List[Any]:
    """Handle AI configuration tasks based on the given configuration and task."""
    # List of functions to call
    config_functions = [
        manage_ai_name,
        manage_ai_role,
        manage_ai_goals,
        manage_plugins,
        manage_api_budget,
    ]

    # Call each function (+ tmp_cfg and task), and store the results in a list
    return [func(config, tmp_cfg, task) for func in config_functions]


def handle_config(
    config: Config, tmp_cfg: Optional[AIConfig], task: str
) -> Optional[AIConfig]:
    """
    Handle the creation or editing of an AI configuration.

    Parameters:
        tmp_cfg (Optional[AIConfig]): The existing AI configuration (optional).
        task (str): The task to perform, either "create" or "edit".

    Returns:
        Optional[AIConfig]: The new or edited AI configuration, or None if an error occurred or the user chose to return to the main menu.
    """
    if tmp_cfg is None:
        tmp_cfg = AIConfig()
    try:
        original_ai_name = None

        if task == "edit" and not tmp_cfg:
            # User wants to edit a tmp_cfg, but no tmp_cfg was provided, so we need to prompt them to select one
            all_configs, _ = AIConfig.load_all(
                config.ai_settings_filepath
            )  # , _ to ignore returned message from load_all
            logger.typewriter_log(
                "Select a configuration to edit:", Fore.GREEN, speak_text=True
            )
            for i, cfg_name in enumerate(all_configs.keys(), start=1):
                logger.typewriter_log(f"{i}. {cfg_name}")
            selection = get_valid_input(config, len(all_configs))
            tmp_cfg = all_configs[list(all_configs.keys())[selection - 1]]
            original_ai_name = tmp_cfg.ai_name

        ai_name, ai_role, ai_goals, plugins, api_budget = handle_configs(
            config, tmp_cfg, task
        )

        new_config = AIConfig(ai_name, ai_role, ai_goals, api_budget, plugins)

        # Save the new or edited configuration
        try:
            new_config.save(
                config.ai_settings_filepath, append=True, old_ai_name=original_ai_name
            )
        except ValueError as e:
            logger.typewriter_log(f"An error occurred: {e}", Fore.RED)
        else:
            logger.typewriter_log("Configuration saved.", Fore.GREEN, speak_text=True)

        # Start up this agent or return to main menu
        user_choice = input(
            f"[E]nter to startup agent: {ai_name} or [R]eturn to main menu: "
        )

        if user_choice.lower() == "e" or user_choice.lower() == "":
            start_prompt(config, new_config, sad=True)

        if user_choice.lower() == "r":
            return None

    except ValueError as e:
        logger.typewriter_log(
            f"Something went wrong: {e}\nReturning to main menu.",
            Fore.RED,
        )

    return new_config
