"""The application entry point.  Can be invoked by a CLI or any other front end application."""
import logging
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Callable, Optional

from colorama import Fore, Style

from autogpt.agents import Agent
from autogpt.config import ConfigBuilder, check_openai_api_key, Config, AIConfig
from autogpt.configurator import create_config
from autogpt.logs import Logger, logger, print_assistant_thoughts, remove_ansi_escape
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT, construct_main_ai_config
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

COMMAND_CATEGORIES = [
    "autogpt.commands.execute_code",
    "autogpt.commands.file_operations",
    "autogpt.commands.web_search",
    "autogpt.commands.web_selenium",
    "autogpt.commands.task_statuses",
]


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
    workspace_directory: str | Path,
    install_plugin_deps: bool,
    ai_name: Optional[str] = None,
    ai_role: Optional[str] = None,
    ai_goals: tuple[str] = tuple(),
):
    # Configure logging before we do anything else.
    logger.set_level(logging.DEBUG if debug else logging.INFO)

    config = ConfigBuilder.build_config_from_env()
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
    workspace_directory = Workspace.get_workspace_directory(config, workspace_directory)

    # HACK: doing this here to collect some globals that depend on the workspace.
    Workspace.build_file_logger_path(config, workspace_directory)

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
        workspace_directory=workspace_directory,
        ai_config=ai_config,
        config=config,
    )




def graceful_agent_interrupt(
    agent: Agent,
    logger_: Logger,
) -> Callable[[int, Optional[FrameType]], None]:
    """Create a signal handler to interrupt an agent executing multiple steps."""

    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        if agent.cycle_budget == 0 or agent.cycles_remaining == 0:
            sys.exit()
        else:
            logger_.typewriter_log(
                "Interrupt signal received. Stopping continuous command execution.",
                Fore.RED,
            )
            agent.cycles_remaining = 0

    return signal_handler


def run_interaction_loop(
    agent: Agent,
):
    config = agent.config
    ai_config = agent.ai_config

    #########################
    # Application Main Loop #
    #########################

    # Set up an interrupt signal for the agent.
    signal.signal(signal.SIGINT, graceful_agent_interrupt(agent, logger))

    if config.debug_mode:
        logger.typewriter_log(
            f"{ai_config.ai_name} System Prompt:", Fore.GREEN, agent.system_prompt
        )

    if config.continuous_mode:
        if config.continuous_limit:
            cycle_budget = config.continuous_limit
        else:
            cycle_budget = None
    else:
        cycle_budget = 1

    cycles_remaining = cycle_budget

    while True:
        if cycles_remaining is not None and cycles_remaining < 0:
            break

        logger.debug(f"Cycle budget: {cycle_budget}; remaining: {cycles_remaining}")

        prompt = agent.on_before_think(DEFAULT_TRIGGERING_PROMPT)

        with Spinner("Thinking... ", plain_output=config.plain_output):
            command_name, command_args, assistant_reply_dict = agent.think(
                prompt,
                DEFAULT_TRIGGERING_PROMPT,
            )

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
            if cycles_remaining is not None:
                cycles_remaining -= 1
        elif command_name.lower().startswith("error"):
            logger.typewriter_log(
                "ERROR: ",
                Fore.RED,
                f"The Agent failed to select an action. "
                f"Error message: {command_name}",
            )
        else:
            logger.typewriter_log(
                "NO ACTION SELECTED: ",
                Fore.RED,
                f"The Agent failed to select an action.",
            )

        user_input = ""
        if cycles_remaining == 0:
            # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
            # Get key press: Prompt the user to press enter to continue or escape
            # to exit
            logger.info(
                f"Enter '{config.authorise_key}' to authorise command, "
                f"'{config.authorise_key} -N' to run N continuous commands, "
                f"'{config.exit_key}' to exit program, or enter feedback for "
                f"{ai_config.ai_name}..."
            )
            while not user_input:
                # Get input from user
                if config.chat_messages_enabled:
                    console_input = clean_input(config, "Waiting for your response...")
                else:
                    console_input = clean_input(
                        config, Fore.MAGENTA + "Input:" + Style.RESET_ALL
                    )

                # Parse user input
                if console_input.lower().strip() == config.authorise_key:
                    user_input = "GENERATE NEXT COMMAND JSON"
                    # Case 1: Continuous iteration was interrupted -> resume
                    # Case 2: The agent used up its cycle budget -> reset
                    cycles_remaining = cycle_budget
                elif console_input.lower().strip() == "":
                    logger.warn("Invalid input format.")
                elif console_input.lower().startswith(f"{config.authorise_key} -"):
                    try:
                        new_cycle_budget = abs(int(console_input.split(" ")[1]))
                        cycle_budget = cycles_remaining = new_cycle_budget
                        user_input = "GENERATE NEXT COMMAND JSON"
                    except ValueError:
                        logger.warn(
                            f"Invalid input format. "
                            f"Please enter '{config.authorise_key} -N'"
                            " where N is the number of continuous tasks."
                        )
                elif console_input.lower() == config.exit_key:
                    user_input = "EXIT"
                else:
                    user_input = console_input
                    command_name = "human_feedback"
                    if cycles_remaining is not None:
                        cycles_remaining += 1

            if user_input == "GENERATE NEXT COMMAND JSON":
                logger.typewriter_log(
                    "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                    Fore.MAGENTA,
                    "",
                )
            elif user_input == "EXIT":
                logger.info("Exiting...")
                exit()
        else:
            # First log new-line so user can differentiate sections better in console
            logger.typewriter_log("\n")
            if cycles_remaining is not None:
                # Print authorized commands left value
                logger.typewriter_log(
                    "AUTHORISED COMMANDS LEFT: ", Fore.CYAN, f"{cycles_remaining}"
                )

        result = agent.execute(command_name, command_args, user_input)

        # history
        if result is not None:
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
        else:
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")
