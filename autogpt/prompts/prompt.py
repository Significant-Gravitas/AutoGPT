from colorama import Fore, Style
from autogpt import utils
from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.config.prompt_config import PromptConfig
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import logger
from autogpt.prompts.generator import PromptGenerator
from autogpt.setup import generate_aiconfig_automatic
from autogpt.utils import clean_input

cfg = Config()

DEFAULT_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)


# Helper functions
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
    prompt_config = PromptConfig(cfg.prompt_settings_file)

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


def get_valid_input(num_options):
    while True:
        try:
            selection = int(clean_input("Please choose a number: "))
            if (
                1 <= selection <= num_options
            ):  # The number must be within the valid range
                return selection
        except ValueError:
            logger.typewriter_log("Invalid input. Please enter a number.", Fore.RED)


def select_config_action(all_configs, task):
    cfg_count = len(all_configs)

    logger.typewriter_log(f"Please select a configuration to {task}:", Fore.GREEN, speak_text=True)
    for i, cfg_name in enumerate(all_configs.keys(), start=1):
        logger.typewriter_log(f"{i}. {cfg_name}")
    logger.typewriter_log(f"{cfg_count+1}. Go back to main menu")
    
    selection = get_valid_input(cfg_count + 1)
    if selection == cfg_count + 1:
        return None
    else:
        return all_configs[list(all_configs.keys())[selection - 1]]


def prompt_for_num_goals(task, current_goal_count=None):
    num_goals = -1
    prompt_message = {
        "create": f"{Fore.GREEN}Set number of goals (0-20): {Style.RESET_ALL}",
        "edit": f"{Fore.GREEN}AI has {current_goal_count} goals. Change? (0-20): {Style.RESET_ALL}",
    }[task]

    fallback_message = {
        "create": f"{Fore.LIGHTBLUE_EX}No input, defaulting to 5 goals.{Style.RESET_ALL}",
        "edit": f"{Fore.LIGHTBLUE_EX}No change, {current_goal_count} goals.{Style.RESET_ALL}",
    }[task]

    while not 0 <= num_goals <= 20:
        num_goals_input = clean_input(prompt_message)

        if num_goals_input == "":
            num_goals = current_goal_count if task == "edit" else 5
            logger.typewriter_log(fallback_message)
        elif num_goals_input.isdigit():
            num_goals = int(num_goals_input)
            if num_goals > 20:
                logger.typewriter_log(
                    "Over 20 goals? Please set manually in settings file.", Fore.RED
                )
            elif num_goals == 0:
                num_goals = 5
                logger.typewriter_log(
                    "No goals set, using default of 5.", Fore.LIGHTBLUE_EX
                )
        else:
            logger.typewriter_log("Invalid input, enter a number (0-20).", Fore.RED)

    return num_goals


def check_ai_name_exists(name):
    """Check if a name exists in the current configurations."""
    all_configs = AIConfig.load_all(cfg.ai_settings_filepath)
    for config in all_configs.values():
        if config.ai_name == name:
            return True
    return False


def generate_unique_name(base):
    """Generate unique AI name."""
    i = 1
    while check_ai_name_exists(f"{base}-{i}"):
        i += 1
    return f"{base}-{i}"


def validate_input(prompt_text):
    while True:
        user_input = input(prompt_text).strip()
        if not user_input.startswith("-"):
            return prompt_text
        print("Input cannot start with '-' or '--'. Please try again.")


# Start AI prompts
def welcome_prompt():
    ai_name = ""
    ai_config = None

    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        f"{Fore.YELLOW}run with '--help' for more information.{Style.RESET_ALL}",
        speak_text=True,
    )

    # Get user desire
    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        f"{Fore.YELLOW}input '--manual' to enter manual mode.{Style.RESET_ALL}",
        speak_text=True,
    )

    user_desire = utils.clean_input(
        f"{Fore.WHITE}I want Auto-GPT to{Style.RESET_ALL}: "
    )

    if user_desire == "":
        user_desire = DEFAULT_USER_DESIRE_PROMPT  # Default prompt

    # If user desire contains "--manual"
    if "--manual" in user_desire:
        logger.typewriter_log(
            "Manual Mode Selected.",
            Fore.LIGHTBLUE_EX,
            speak_text=True,
        )
        return handle_config(None, "create")

    else:
        try:
            return generate_aiconfig_automatic(user_desire)
        except Exception as e:
            logger.typewriter_log(
                "Unable to automatically generate AI Config based on user desire.",
                Fore.RED,
                f"{Fore.LIGHTBLUE_EX}Falling back to manual mode.{Style.RESET_ALL}",
                speak_text=True,
            )

            return handle_config(None, "create")


def start_prompt(config, sad=None):
    from autogpt.main import start_agent_directly

    if cfg.skip_reprompt and config.ai_name:
        logger.typewriter_log("Name :", Fore.GREEN, config.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, config.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{config.ai_goals}")
        logger.typewriter_log(
            "Budget:",
            Fore.GREEN,
            "infinite" if config.api_budget <= 0 else f"${str(config.api_budget)}",
        )

    if cfg.restrict_to_workspace:
        logger.typewriter_log(
            f"{Fore.YELLOW}NOTE:All files/directories created by this agent can be found inside its workspace at:{Style.RESET_ALL}"
        )
        logger.typewriter_log(f"{Fore.YELLOW}-{Style.RESET_ALL}   {cfg.workspace_path}")
    # Set total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(config.api_budget)

    # Agent Created
    logger.typewriter_log(
        f"{Fore.LIGHTBLUE_EX}Auto-GPT has started with the following details:{Style.RESET_ALL}",
        speak_text=True,
    )

    logger.typewriter_log("Name:", Fore.GREEN, config.ai_name, speak_text=False)
    logger.typewriter_log("Role:", Fore.GREEN, config.ai_role, speak_text=False)
    logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
    for goal in config.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)
    logger.typewriter_log(
        "Budget:", Fore.GREEN, f"${str(config.api_budget)}", speak_text=False
    )
    if hasattr(config, "plugins"):
        logger.typewriter_log("Plugins:", Fore.GREEN, "", speak_text=False)
        for plugin in config.plugins:
            logger.typewriter_log("-", Fore.GREEN, plugin, speak_text=False)

    # don't exec when called from main
    if sad:
        start_agent_directly(config, cfg)


# Main menu functions
def display_main_menu(all_configs, menu_options):
    cfg_count = len(all_configs)

    logger.typewriter_log("Please select configuration/option:", Fore.GREEN, speak_text=True)
    for i, cfg_name in enumerate(all_configs.keys(), start=1):
        logger.typewriter_log(f"{i}. {cfg_name}")
    for i, (option_text, _) in enumerate(menu_options, start=cfg_count + 1):
        logger.typewriter_log(f"{i}. {option_text}")


def main_menu() -> AIConfig:
    config = None
    config_file = cfg.ai_settings_filepath

    menu_options = [
        ("Create new configuration", lambda config: handle_config(config, "create")),
        ("Edit existing configuration", lambda config: handle_config(config, "edit")),
        ("View existing configuration", lambda config: view_config(config)),
        ("Delete existing configuration", lambda config: delete_config(config)),
        # add more options here...
    ]

    while True:
        all_configs = AIConfig.load_all(config_file)
        all_configs_keys = list(all_configs.keys())
        cfg_count = len(all_configs)  # Configurations count
        total_options = cfg_count + len(menu_options)

        logger.typewriter_log(
            f"{Fore.GREEN}Attempting to load {cfg.ai_settings_file}: {Fore.YELLOW}{ {True: 'load successful.', False: 'no configuration(s) detected.'}[bool(all_configs)] }{Style.RESET_ALL}"
        )

        if cfg_count > 0:
            logger.typewriter_log(
                "Welcome back! ",
                Fore.GREEN,
                f"{Fore.YELLOW}Configuration(s) detected.{Style.RESET_ALL}",
                speak_text=True,
            )
            display_main_menu(all_configs, menu_options)  # Displaying menu here
            selection = get_valid_input(total_options)

            if 1 <= selection <= cfg_count:  # If selection is a valid index
                config_name = all_configs_keys[selection - 1]
                config = all_configs[config_name]
                return config
            else:  # "Create", "Edit", "View", "Delete"
                task_index = selection - cfg_count - 1
                task = menu_options[task_index][0].split()[0].lower()
                _, action = menu_options[task_index]
                if task in ["edit", "view", "delete"]:
                    chosen_config = select_config_action(all_configs, task)
                    if chosen_config is None:
                        continue  # Start over
                    config = action(chosen_config)
                else:  # On "create"
                    config = AIConfig()
                    config = action(config)
                if config:  # If an action returns a config
                    return config

        else:  # If no configurations exist
            config = AIConfig()
            config = welcome_prompt()
            return config


# Create, edit configurations
def manage_ai_name(configs, task):
    prompts = {
        "create": ("Create an AI-Assistant:", "For example, 'Entrepreneur-GPT'"),
        "edit": (
            "Edit the AI name:",
            f"(current: '{configs.ai_name if configs else None}'), leave empty to keep current name.",
        ),
    }
    logger.typewriter_log(
        prompts[task][0],
        Fore.GREEN,
        f"{Fore.YELLOW}{prompts[task][1]}{Style.RESET_ALL}",
        speak_text=True,
    )

    prompt_text = "AI Name: "

    input_name = utils.clean_input(prompt_text)
    ai_name = input_name if input_name or task == "create" else configs.ai_name
    while check_ai_name_exists(ai_name) and (input_name or task == "create"):
        logger.typewriter_log(
            f"{Fore.RED}This AI name already exists. Please choose a different name.{Style.RESET_ALL}",
            speak_text=True,
        )
        input_name = utils.clean_input(validate_input(prompt_text))
    logger.typewriter_log(f"{Fore.LIGHTBLUE_EX}AI name set.", speak_text=True)
    configs.ai_name = ai_name
    return ai_name


def manage_ai_role(config, task):
    default_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    if task == "create":
        logger.typewriter_log(
            "Describe your AI's role:",
            Fore.GREEN,
            f"{Fore.YELLOW}For example, '{default_role}'{Style.RESET_ALL}",
            speak_text=True,
        )
        current_role = ""
    else:
        current_role = config.ai_role
        logger.typewriter_log(
            "Describe your AI's role:",
            Fore.GREEN,
            f"{Fore.YELLOW}Current: '{current_role}', use [Enter] to keep the current role / save the input.{Style.RESET_ALL}",
            speak_text=True,
        )

    ai_role = (
        utils.clean_input(f"{config.ai_name if config else None} is: ")
        or current_role
        or default_role
    )

    if ai_role == default_role:
        logger.typewriter_log(
            f"{Fore.LIGHTBLUE_EX}No input, defaulting to AI role: '{default_role}'",
            speak_text=True,
        )

    logger.typewriter_log(f"{Fore.LIGHTBLUE_EX}AI role set.", speak_text=True)

    config.ai_role = ai_role
    return ai_role


def manage_ai_goals(config, task):
    default_goals = config.ai_goals if config else []
    max_goals = prompt_for_num_goals(task, len(default_goals))

    ai_goals = []
    num_default_goals = len(default_goals)

    if task == "create":
        logger.typewriter_log(
            f"Enter up to {max_goals} goals for your AI: ", Fore.GREEN
        )

        for i in range(max_goals):
            ai_goal = utils.clean_input(f"{Fore.GREEN}Goal {i+1}:{Style.RESET_ALL} ")
            logger.info(f"{Fore.WHITE}Use [Enter] to save the input.{Style.RESET_ALL}")
            if ai_goal == "":
                break
            ai_goals.append(ai_goal)

    else:
        logger.typewriter_log(f"Enter up to {max_goals} goals for your AI:", Fore.GREEN)
        kept_goals = []

        for i in range(num_default_goals):
            logger.typewriter_log(
                f"{Fore.GREEN}Current Goal {i+1}: {Fore.YELLOW}'{default_goals[i]}'{Style.RESET_ALL}"
            )
            action = utils.clean_input(
                "Do you want to [E]dit, [D]elete, or [K]eep this goal? "
            )
            if action.lower() == "e":
                ai_goal = utils.clean_input(
                    f"{Fore.GREEN}New Goal {i+1}:{Style.RESET_ALL} "
                )
                if ai_goal != "":
                    kept_goals.append(ai_goal)
                else:
                    kept_goals.append(default_goals[i])
            elif action.lower() == "d":
                continue
            else:
                kept_goals.append(default_goals[i])

        ai_goals = kept_goals
        num_ai_goals = len(ai_goals)

        if num_ai_goals < max_goals:
            logger.typewriter_log(
                f"You can add up to {max_goals - num_ai_goals} new goals.", Fore.GREEN
            )
            for i in range(num_ai_goals, max_goals):
                ai_goal = utils.clean_input(
                    f"{Fore.GREEN}New Goal {i+1}:{Style.RESET_ALL} "
                )
                if ai_goal == "":
                    break
                ai_goals.append(ai_goal)
    logger.typewriter_log(f"{Fore.LIGHTBLUE_EX}AI goals set.", speak_text=True)
    config.ai_goals = ai_goals
    return ai_goals


def manage_plugins(config, task):
    default_plugins = cfg.plugins_allowlist if cfg and cfg.plugins_allowlist else []
    if default_plugins:
        logger.typewriter_log("Add plugins from the plugins_allowlist?", Fore.GREEN)

        if task == "create":
            # Select new plugins from allowlist
            env_plugins = list(default_plugins)
            remaining_plugins = []
            for i in range(len(env_plugins)):
                action = utils.clean_input(
                    f"{Fore.GREEN}Plugin {i+1}: {Fore.YELLOW}'{env_plugins[i]}'\n{Fore.WHITE}Do you want to [I]gnore or [A]dd this plugin?{Style.RESET_ALL} "
                )
                if action.lower() == "a":
                    remaining_plugins.append(env_plugins[i])

            plugins = remaining_plugins
        else:
            # Edit existing plugins
            plugins = config.plugins
            deleted_plugins = []  # To keep track of deleted plugins
            i = 0
            while i < len(plugins):
                logger.typewriter_log(
                    f"{Fore.GREEN}Current plugin {i+1}: {Fore.YELLOW}'{plugins[i]}'{Style.RESET_ALL}",
                    Fore.LIGHTBLUE_EX,
                    speak_text=False,
                )
                action = utils.clean_input(
                    f"{Fore.WHITE}Do you want to [D]elete or [K]eep this plugin?{Style.RESET_ALL} "
                )
                if action.lower() == "d":
                    deleted_plugins.append(plugins[i])  # Keep track of deleted plugin
                    del plugins[i]
                else:
                    i += 1

            # Offer to add plugins from allowlist
            env_plugins = default_plugins
            for plugin in env_plugins:
                if (
                    plugin not in plugins and plugin not in deleted_plugins
                ):  # Check if the plugin was not deleted
                    logger.typewriter_log(
                        f"{Fore.GREEN}New plugin available: {Fore.YELLOW}'{plugin}'{Style.RESET_ALL}",
                        Fore.LIGHTBLUE_EX,
                        speak_text=False,
                    )
                    action = utils.clean_input(
                        f"{Fore.WHITE}Do you want to ignore [E]nter or [A]dd this plugin?{Style.RESET_ALL} "
                    )
                    if action.lower() == "a":
                        plugins.append(plugin)

    config.plugins = plugins
    return plugins


def manage_api_budget(config, task):
    default_budget = config.api_budget if config else 0.0
    if task == "create":
        logger.typewriter_log(
            f"{Fore.GREEN}Enter your budget for API calls: {Fore.YELLOW}For example: $1.50, leave empty for unlimited budget.{Style.RESET_ALL}",
            speak_text=True,
        )
    else:
        logger.typewriter_log(
            f"{Fore.GREEN}Enter your budget for API calls:\n{Fore.YELLOW}Current: '${default_budget}'. For example: $1.50, leave empty to keep current budget.{Style.RESET_ALL}",
            speak_text=True,
        )

    logger.info("Use [Enter] to save the input.")
    api_budget_input = utils.clean_input(f"{Fore.WHITE}Budget: ${Style.RESET_ALL}")
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

    config.api_budget = api_budget
    return api_budget


# View, delete configurations
def view_config(config):
    logger.typewriter_log(
        f"{Fore.LIGHTBLUE_EX}Configuration {config.ai_name} has the following details:{Style.RESET_ALL}",
        speak_text=True,
    )
    logger.typewriter_log("Name:", Fore.GREEN, config.ai_name, speak_text=False)
    logger.typewriter_log("Role:", Fore.GREEN, config.ai_role, speak_text=False)
    logger.typewriter_log("Goals:", Fore.GREEN, "", speak_text=False)
    for goal in config.ai_goals:
        logger.typewriter_log("-", Fore.GREEN, goal, speak_text=False)
    logger.typewriter_log(
        "Budget:", Fore.GREEN, f"${str(config.api_budget)}", speak_text=False
    )
    if hasattr(config, "plugins"):
        logger.typewriter_log("Plugins:", Fore.GREEN, "", speak_text=False)
        for plugin in config.plugins:
            logger.typewriter_log("-", Fore.GREEN, plugin, speak_text=False)

    # Prompt
    choice = input("Do you want to [E]dit or [R]eturn to the main menu? ").lower()

    if choice == "e":
        return handle_config(config, "edit")

    logger.typewriter_log("Returning to the main menu.", speak_text=True)
    return main_menu()


def delete_config(selected_config):
    # Delete configuration

    logger.typewriter_log(
        f"Deleting the configuration for: {selected_config.ai_name}.",
        Fore.RED,
    )
    AIConfig().delete(
        config_file=cfg.ai_settings_filepath,
        ai_name=selected_config.ai_name,
    )
    return main_menu()


# Menu handling
def handle_configs(config, task):
    # List of functions to call
    config_functions = [
        manage_ai_name,
        manage_ai_role,
        manage_ai_goals,
        manage_plugins,
        manage_api_budget,
    ]

    # Call each function with config and task, and store the results in a list
    return [func(config, task) for func in config_functions]


def handle_config(config, task):
    try:
        original_ai_name = None

        if task == "edit" and not config:
            # User wants to edit a config, but no config was provided, so we need to prompt them to select one
            all_configs = AIConfig.load_all(cfg.ai_settings_filepath)
            logger.typewriter_log(
                "Please select a configuration to edit:", Fore.GREEN, speak_text=True
            )
            for i, cfg_name in enumerate(all_configs.keys(), start=1):
                logger.typewriter_log(f"{i}. {cfg_name}")
            selection = get_valid_input(len(all_configs))
            config = all_configs[all_configs.keys()[selection - 1]]
            original_ai_name = config.ai_name

        # For create, config is None, so the config will be created from scratch in handle_configs
        ai_name, ai_role, ai_goals, plugins, api_budget = handle_configs(config, task)

        new_config = AIConfig(ai_name, ai_role, ai_goals, api_budget, plugins)

        # Save the new or edited configuration
        new_config.save(
            cfg.ai_settings_filepath, append=True, old_ai_name=original_ai_name
        )

        logger.typewriter_log("Configuration saved.", Fore.GREEN)

        # Ask user to start up this agent or return to main menu
        user_choice = input(
            f"[E]nter to startup agent: {ai_name} or [R]eturn to main menu: "
        )

        if user_choice.lower() == "e" or user_choice.lower() == "":
            return start_prompt(new_config, sad=True)

    except ValueError as e:
        logger.typewriter_log(
            f"Something went wrong: {e}\nReturning to main menu.",
            Fore.RED,
        )
        return main_menu()
