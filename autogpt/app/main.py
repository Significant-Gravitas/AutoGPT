"""The application entry point.  Can be invoked by a CLI or any other front end application."""
import enum
import logging
import math
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Optional

from colorama import Fore, Style

from autogpt.agents import Agent, AgentThoughts, CommandArgs, CommandName
from autogpt.app.configurator import create_config
from autogpt.app.setup import prompt_user
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, Config, ConfigBuilder, check_openai_api_key
from autogpt.llm.api_manager import ApiManager
from autogpt.logs import logger
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import (
    clean_input,
    get_current_git_branch,
    get_latest_bulletin,
    get_legal_warning,
    markdown_to_ansi_style,
)
from autogpt.workspace import Workspace
from scripts.install_plugin_deps import install_plugin_dependencies


def run_auto_gpt(
    continuous: bool,
    continuous_limit: int,
    ai_settings: str,
    prompt_settings: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    working_directory: Path,
    workspace_directory: str | Path,
    install_plugin_deps: bool,
    ai_name: Optional[str] = None,
    ai_role: Optional[str] = None,
    ai_goals: tuple[str] = tuple(),
):
    # Configure logging before we do anything else.
    logger.set_level(logging.DEBUG if debug else logging.INFO)

    config = ConfigBuilder.build_config_from_env(workdir=working_directory)

    # HACK: This is a hack to allow the config into the logger without having to pass it around everywhere
    # or import it directly.
    logger.config = config

    # TODO: fill in llm values here
    check_openai_api_key(config)

    create_config(
        config,
        continuous,
        continuous_limit,
        ai_settings,
        prompt_settings,
        skip_reprompt,
        speak,
        debug,
        gpt3only,
        gpt4only,
        memory_type,
        browser_name,
        allow_downloads,
        skip_news,
    )

    if config.continuous_mode:
        for line in get_legal_warning().split("\n"):
            logger.warn(markdown_to_ansi_style(line), "LEGAL:", Fore.RED)

    if not config.skip_news:
        motd, is_new_motd = get_latest_bulletin()
        if motd:
            motd = markdown_to_ansi_style(motd)
            for motd_line in motd.split("\n"):
                logger.info(motd_line, "NEWS:", Fore.GREEN)
            if is_new_motd and not config.chat_messages_enabled:
                input(
                    Fore.MAGENTA
                    + Style.BRIGHT
                    + "NEWS: Bulletin was updated! Press Enter to continue..."
                    + Style.RESET_ALL
                )

        git_branch = get_current_git_branch()
        if git_branch and git_branch != "stable":
            logger.typewriter_log(
                "WARNING: ",
                Fore.RED,
                f"You are running on `{git_branch}` branch "
                "- this is not a supported branch.",
            )
        if sys.version_info < (3, 10):
            logger.typewriter_log(
                "WARNING: ",
                Fore.RED,
                "You are running on an older version of Python. "
                "Some people have observed problems with certain "
                "parts of Auto-GPT with this version. "
                "Please consider upgrading to Python 3.10 or higher.",
            )

    if install_plugin_deps:
        install_plugin_dependencies()

    # TODO: have this directory live outside the repository (e.g. in a user's
    #   home directory) and have it come in as a command line argument or part of
    #   the env file.
    Workspace.set_workspace_directory(config, workspace_directory)

    # HACK: doing this here to collect some globals that depend on the workspace.
    Workspace.set_file_logger_path(config, config.workspace_path)

    config.plugins = scan_plugins(config, config.debug_mode)
    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry()

    logger.debug(
        f"The following command categories are disabled: {config.disabled_command_categories}"
    )
    enabled_command_categories = [
        x for x in COMMAND_CATEGORIES if x not in config.disabled_command_categories
    ]

    logger.debug(
        f"The following command categories are enabled: {enabled_command_categories}"
    )

    for command_category in enabled_command_categories:
        command_registry.import_commands(command_category)

    # Unregister commands that are incompatible with the current config
    incompatible_commands = []
    for command in command_registry.commands.values():
        if callable(command.enabled) and not command.enabled(config):
            command.enabled = False
            incompatible_commands.append(command)

    for command in incompatible_commands:
        command_registry.unregister(command)
        logger.debug(
            f"Unregistering incompatible command: {command.name}, "
            f"reason - {command.disabled_reason or 'Disabled by current config.'}"
        )

    ai_config = construct_main_ai_config(
        config,
        name=ai_name,
        role=ai_role,
        goals=ai_goals,
    )
    ai_config.command_registry = command_registry
    # print(prompt)

    # add chat plugins capable of report to logger
    if config.chat_messages_enabled:
        for plugin in config.plugins:
            if hasattr(plugin, "can_handle_report") and plugin.can_handle_report():
                logger.info(f"Loaded plugin into logger: {plugin.__class__.__name__}")
                logger.chat_plugins.append(plugin)

    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(config)
    memory.clear()
    logger.typewriter_log(
        "Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log("Using Browser:", Fore.GREEN, config.selenium_web_browser)

    agent = Agent(
        memory=memory,
        command_registry=command_registry,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        ai_config=ai_config,
        config=config,
    )

    run_interaction_loop(agent)


def _get_cycle_budget(continuous_mode: bool, continuous_limit: int) -> int | None:
    # Translate from the continuous_mode/continuous_limit config
    # to a cycle_budget (maximum number of cycles to run without checking in with the
    # user) and a count of cycles_remaining before we check in..
    if continuous_mode:
        cycle_budget = continuous_limit if continuous_limit else math.inf
    else:
        cycle_budget = 1

    return cycle_budget


class UserFeedback(str, enum.Enum):
    """Enum for user feedback."""

    AUTHORIZE = "GENERATE NEXT COMMAND JSON"
    EXIT = "EXIT"
    TEXT = "TEXT"


def run_interaction_loop(
    agent: Agent,
) -> None:
    """Run the main interaction loop for the agent.

    Args:
        agent: The agent to run the interaction loop for.

    Returns:
        None
    """
    # These contain both application config and agent config, so grab them here.
    config = agent.config
    ai_config = agent.ai_config
    logger.debug(f"{ai_config.ai_name} System Prompt: {agent.system_prompt}")

    cycle_budget = cycles_remaining = _get_cycle_budget(
        config.continuous_mode, config.continuous_limit
    )
    spinner = Spinner("Thinking...", plain_output=config.plain_output)

    def graceful_agent_interrupt(signum: int, frame: Optional[FrameType]) -> None:
        nonlocal cycle_budget, cycles_remaining, spinner
        if cycles_remaining in [0, 1, math.inf]:
            logger.typewriter_log(
                "Interrupt signal received. Stopping continuous command execution "
                "immediately.",
                Fore.RED,
            )
            sys.exit()
        else:
            restart_spinner = spinner.running
            if spinner.running:
                spinner.stop()

            logger.typewriter_log(
                "Interrupt signal received. Stopping continuous command execution.",
                Fore.RED,
            )
            cycles_remaining = 1
            if restart_spinner:
                spinner.start()

    # Set up an interrupt signal for the agent.
    signal.signal(signal.SIGINT, graceful_agent_interrupt)

    #########################
    # Application Main Loop #
    #########################

    while cycles_remaining > 0:
        logger.debug(f"Cycle budget: {cycle_budget}; remaining: {cycles_remaining}")

        ########
        # Plan #
        ########
        # Have the agent determine the next action to take.
        with spinner:
            command_name, command_args, assistant_reply_dict = agent.think()

        ###############
        # Update User #
        ###############
        # Print the assistant's thoughts and the next command to the user.
        update_user(config, ai_config, command_name, command_args, assistant_reply_dict)

        ##################
        # Get user input #
        ##################
        if cycles_remaining == 1:  # Last cycle
            user_feedback, user_input, new_cycles_remaining = get_user_feedback(
                config,
                ai_config,
            )

            if user_feedback == UserFeedback.AUTHORIZE:
                if new_cycles_remaining is not None:
                    # Case 1: User is altering the cycle budget.
                    if cycle_budget > 1:
                        cycle_budget = new_cycles_remaining + 1
                    # Case 2: User is running iteratively and
                    #   has initiated a one-time continuous cycle
                    cycles_remaining = new_cycles_remaining + 1
                else:
                    # Case 1: Continuous iteration was interrupted -> resume
                    if cycle_budget > 1:
                        logger.typewriter_log(
                            "RESUMING CONTINUOUS EXECUTION: ",
                            Fore.MAGENTA,
                            f"The cycle budget is {cycle_budget}.",
                        )
                    # Case 2: The agent used up its cycle budget -> reset
                    cycles_remaining = cycle_budget + 1
                logger.typewriter_log(
                    "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                    Fore.MAGENTA,
                    "",
                )
            elif user_feedback == UserFeedback.EXIT:
                logger.typewriter_log("Exiting...", Fore.YELLOW)
                exit()
            else:  # user_feedback == UserFeedback.TEXT
                command_name = "human_feedback"
        else:
            user_input = None
            # First log new-line so user can differentiate sections better in console
            logger.typewriter_log("\n")
            if cycles_remaining != math.inf:
                # Print authorized commands left value
                logger.typewriter_log(
                    "AUTHORISED COMMANDS LEFT: ", Fore.CYAN, f"{cycles_remaining}"
                )

        ###################
        # Execute Command #
        ###################
        # Decrement the cycle counter first to reduce the likelihood of a SIGINT
        # happening during command execution, setting the cycles remaining to 1,
        # and then having the decrement set it to 0, exiting the application.
        if command_name != "human_feedback":
            cycles_remaining -= 1
        result = agent.execute(command_name, command_args, user_input)

        if result is not None:
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
        else:
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")


def update_user(
    config: Config,
    ai_config: AIConfig,
    command_name: CommandName | None,
    command_args: CommandArgs | None,
    assistant_reply_dict: AgentThoughts,
) -> None:
    """Prints the assistant's thoughts and the next command to the user.

    Args:
        config: The program's configuration.
        ai_config: The AI's configuration.
        command_name: The name of the command to execute.
        command_args: The arguments for the command.
        assistant_reply_dict: The assistant's reply.
    """

    print_assistant_thoughts(ai_config.ai_name, assistant_reply_dict, config)

    if command_name is not None:
        if config.speak_mode:
            say_text(f"I want to execute {command_name}", config)

        # First log new-line so user can differentiate sections better in console
        logger.typewriter_log("\n")
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{remove_ansi_escape(command_name)}{Style.RESET_ALL}  "
            f"ARGUMENTS = {Fore.CYAN}{command_args}{Style.RESET_ALL}",
        )
    elif command_name.lower().startswith("error"):
        logger.typewriter_log(
            "ERROR: ",
            Fore.RED,
            f"The Agent failed to select an action. " f"Error message: {command_name}",
        )
    else:
        logger.typewriter_log(
            "NO ACTION SELECTED: ",
            Fore.RED,
            f"The Agent failed to select an action.",
        )


def get_user_feedback(
    config: Config,
    ai_config: AIConfig,
) -> tuple[UserFeedback, str, int | None]:
    """Gets the user's feedback on the assistant's reply.

    Args:
        config: The program's configuration.
        ai_config: The AI's configuration.

    Returns:
        A tuple of the user's feedback, the user's input, and the number of
        cycles remaining if the user has initiated a continuous cycle.
    """
    # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
    # Get key press: Prompt the user to press enter to continue or escape
    # to exit
    logger.info(
        f"Enter '{config.authorise_key}' to authorise command, "
        f"'{config.authorise_key} -N' to run N continuous commands, "
        f"'{config.exit_key}' to exit program, or enter feedback for "
        f"{ai_config.ai_name}..."
    )

    user_feedback = None
    user_input = ""
    new_cycles_remaining = None

    while user_feedback is None:
        # Get input from user
        if config.chat_messages_enabled:
            console_input = clean_input(config, "Waiting for your response...")
        else:
            console_input = clean_input(
                config, Fore.MAGENTA + "Input:" + Style.RESET_ALL
            )

        # Parse user input
        if console_input.lower().strip() == config.authorise_key:
            user_feedback = UserFeedback.AUTHORIZE
        elif console_input.lower().strip() == "":
            logger.warn("Invalid input format.")
        elif console_input.lower().startswith(f"{config.authorise_key} -"):
            try:
                user_feedback = UserFeedback.AUTHORIZE
                new_cycles_remaining = abs(int(console_input.split(" ")[1]))
            except ValueError:
                logger.warn(
                    f"Invalid input format. "
                    f"Please enter '{config.authorise_key} -N'"
                    " where N is the number of continuous tasks."
                )
        elif console_input.lower() in [config.exit_key, "exit"]:
            user_feedback = UserFeedback.EXIT
        else:
            user_feedback = UserFeedback.TEXT
            user_input = console_input

    return user_feedback, user_input, new_cycles_remaining


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
    ai_config = AIConfig.load(config.workdir / config.ai_settings_file)

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
        ai_config.save(config.workdir / config.ai_settings_file)

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


def print_assistant_thoughts(
    ai_name: str,
    assistant_reply_json_valid: dict,
    config: Config,
) -> None:
    from autogpt.speech import say_text

    assistant_thoughts_reasoning = None
    assistant_thoughts_plan = None
    assistant_thoughts_speak = None
    assistant_thoughts_criticism = None

    assistant_thoughts = assistant_reply_json_valid.get("thoughts", {})
    assistant_thoughts_text = remove_ansi_escape(assistant_thoughts.get("text", ""))
    if assistant_thoughts:
        assistant_thoughts_reasoning = remove_ansi_escape(
            assistant_thoughts.get("reasoning", "")
        )
        assistant_thoughts_plan = remove_ansi_escape(assistant_thoughts.get("plan", ""))
        assistant_thoughts_criticism = remove_ansi_escape(
            assistant_thoughts.get("criticism", "")
        )
        assistant_thoughts_speak = remove_ansi_escape(
            assistant_thoughts.get("speak", "")
        )
    logger.typewriter_log(
        f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text
    )
    logger.typewriter_log("REASONING:", Fore.YELLOW, str(assistant_thoughts_reasoning))
    if assistant_thoughts_plan:
        logger.typewriter_log("PLAN:", Fore.YELLOW, "")
        # If it's a list, join it into a string
        if isinstance(assistant_thoughts_plan, list):
            assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
        elif isinstance(assistant_thoughts_plan, dict):
            assistant_thoughts_plan = str(assistant_thoughts_plan)

        # Split the input_string using the newline character and dashes
        lines = assistant_thoughts_plan.split("\n")
        for line in lines:
            line = line.lstrip("- ")
            logger.typewriter_log("- ", Fore.GREEN, line.strip())
    logger.typewriter_log("CRITICISM:", Fore.YELLOW, f"{assistant_thoughts_criticism}")
    # Speak the assistant's thoughts
    if assistant_thoughts_speak:
        if config.speak_mode:
            say_text(assistant_thoughts_speak, config)
        else:
            logger.typewriter_log("SPEAK:", Fore.YELLOW, f"{assistant_thoughts_speak}")


def remove_ansi_escape(s: str) -> str:
    return s.replace("\x1B", "")
