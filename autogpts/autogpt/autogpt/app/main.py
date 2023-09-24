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
from pydantic import SecretStr

from autogpt.agents import AgentThoughts, CommandArgs, CommandName
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.agents.utils.exceptions import InvalidAgentResponseError
from autogpt.app.configurator import create_config
from autogpt.app.setup import interactive_ai_config_setup
from autogpt.app.spinner import Spinner
from autogpt.app.utils import (
    clean_input,
    get_current_git_branch,
    get_latest_bulletin,
    get_legal_warning,
    markdown_to_ansi_style,
)
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIConfig, Config, ConfigBuilder, check_openai_api_key
from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    ModelProviderCredentials,
)
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from autogpt.core.runner.client_lib.utils import coroutine
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.config import configure_chat_plugins, configure_logging
from autogpt.logs.helpers import print_attribute, speak
from autogpt.memory.vector import get_memory
from autogpt.models.command_registry import CommandRegistry
from autogpt.plugins import scan_plugins
from autogpt.workspace import Workspace
from scripts.install_plugin_deps import install_plugin_dependencies


@coroutine
async def run_auto_gpt(
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
    config = ConfigBuilder.build_config_from_env(workdir=working_directory)

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

    # Set up logging module
    configure_logging(config)

    llm_provider = _configure_openai_provider(config)

    logger = logging.getLogger(__name__)

    if config.continuous_mode:
        for line in get_legal_warning().split("\n"):
            logger.warn(
                extra={
                    "title": "LEGAL:",
                    "title_color": Fore.RED,
                    "preserve_color": True,
                },
                msg=markdown_to_ansi_style(line),
            )

    if not config.skip_news:
        motd, is_new_motd = get_latest_bulletin()
        if motd:
            motd = markdown_to_ansi_style(motd)
            for motd_line in motd.split("\n"):
                logger.info(
                    extra={
                        "title": "NEWS:",
                        "title_color": Fore.GREEN,
                        "preserve_color": True,
                    },
                    msg=motd_line,
                )
            if is_new_motd and not config.chat_messages_enabled:
                input(
                    Fore.MAGENTA
                    + Style.BRIGHT
                    + "NEWS: Bulletin was updated! Press Enter to continue..."
                    + Style.RESET_ALL
                )

        git_branch = get_current_git_branch()
        if git_branch and git_branch != "stable":
            logger.warn(
                f"You are running on `{git_branch}` branch"
                " - this is not a supported branch."
            )
        if sys.version_info < (3, 10):
            logger.error(
                "WARNING: You are running on an older version of Python. "
                "Some people have observed problems with certain "
                "parts of AutoGPT with this version. "
                "Please consider upgrading to Python 3.10 or higher.",
            )

    if install_plugin_deps:
        install_plugin_dependencies()

    # TODO: have this directory live outside the repository (e.g. in a user's
    #   home directory) and have it come in as a command line argument or part of
    #   the env file.
    config.workspace_path = Workspace.init_workspace_directory(
        config, workspace_directory
    )

    # HACK: doing this here to collect some globals that depend on the workspace.
    config.file_logger_path = Workspace.build_file_logger_path(config.workspace_path)

    config.plugins = scan_plugins(config, config.debug_mode)
    configure_chat_plugins(config)

    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)

    ai_config = await construct_main_ai_config(
        config,
        llm_provider=llm_provider,
        name=ai_name,
        role=ai_role,
        goals=ai_goals,
    )
    # print(prompt)

    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(config)
    memory.clear()
    print_attribute("Configured Memory", memory.__class__.__name__)

    print_attribute("Configured Browser", config.selenium_web_browser)

    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_config=ai_config,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        command_registry=command_registry,
        memory=memory,
        legacy_config=config,
    )

    await run_interaction_loop(agent)


def _configure_openai_provider(config: Config) -> OpenAIProvider:
    """Create a configured OpenAIProvider object.

    Args:
        config: The program's configuration.

    Returns:
        A configured OpenAIProvider object.
    """
    if config.openai_api_key is None:
        raise RuntimeError("OpenAI key is not configured")

    openai_settings = OpenAIProvider.default_settings.copy(deep=True)
    openai_settings.credentials = ModelProviderCredentials(
        api_key=SecretStr(config.openai_api_key),
        # TODO: support OpenAI Azure credentials
        api_base=SecretStr(config.openai_api_base) if config.openai_api_base else None,
        api_type=SecretStr(config.openai_api_type) if config.openai_api_type else None,
        api_version=SecretStr(config.openai_api_version)
        if config.openai_api_version
        else None,
    )
    return OpenAIProvider(
        settings=openai_settings,
        logger=logging.getLogger("OpenAIProvider"),
    )


def _get_cycle_budget(continuous_mode: bool, continuous_limit: int) -> int | float:
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


async def run_interaction_loop(
    agent: Agent,
) -> None:
    """Run the main interaction loop for the agent.

    Args:
        agent: The agent to run the interaction loop for.

    Returns:
        None
    """
    # These contain both application config and agent config, so grab them here.
    legacy_config = agent.legacy_config
    ai_config = agent.ai_config
    logger = logging.getLogger(__name__)

    logger.debug(f"{ai_config.ai_name} System Prompt:\n{agent.system_prompt}")

    cycle_budget = cycles_remaining = _get_cycle_budget(
        legacy_config.continuous_mode, legacy_config.continuous_limit
    )
    spinner = Spinner("Thinking...", plain_output=legacy_config.plain_output)

    def graceful_agent_interrupt(signum: int, frame: Optional[FrameType]) -> None:
        nonlocal cycle_budget, cycles_remaining, spinner
        if cycles_remaining in [0, 1]:
            logger.error("Interrupt signal received. Stopping AutoGPT immediately.")
            sys.exit()
        else:
            restart_spinner = spinner.running
            if spinner.running:
                spinner.stop()

            logger.error(
                "Interrupt signal received. Stopping continuous command execution."
            )
            cycles_remaining = 1
            if restart_spinner:
                spinner.start()

    # Set up an interrupt signal for the agent.
    signal.signal(signal.SIGINT, graceful_agent_interrupt)

    #########################
    # Application Main Loop #
    #########################

    # Keep track of consecutive failures of the agent
    consecutive_failures = 0

    while cycles_remaining > 0:
        logger.debug(f"Cycle budget: {cycle_budget}; remaining: {cycles_remaining}")

        ########
        # Plan #
        ########
        # Have the agent determine the next action to take.
        with spinner:
            try:
                command_name, command_args, assistant_reply_dict = await agent.think()
            except InvalidAgentResponseError as e:
                logger.warn(f"The agent's thoughts could not be parsed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.error(
                        f"The agent failed to output valid thoughts {consecutive_failures} "
                        "times in a row. Terminating..."
                    )
                    sys.exit()
                continue

        consecutive_failures = 0

        ###############
        # Update User #
        ###############
        # Print the assistant's thoughts and the next command to the user.
        update_user(
            legacy_config, ai_config, command_name, command_args, assistant_reply_dict
        )

        ##################
        # Get user input #
        ##################
        if cycles_remaining == 1:  # Last cycle
            user_feedback, user_input, new_cycles_remaining = await get_user_feedback(
                legacy_config,
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
                        logger.info(
                            f"The cycle budget is {cycle_budget}.",
                            extra={
                                "title": "RESUMING CONTINUOUS EXECUTION",
                                "title_color": Fore.MAGENTA,
                            },
                        )
                    # Case 2: The agent used up its cycle budget -> reset
                    cycles_remaining = cycle_budget + 1
                logger.info(
                    "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                    extra={"color": Fore.MAGENTA},
                )
            elif user_feedback == UserFeedback.EXIT:
                logger.warn("Exiting...")
                exit()
            else:  # user_feedback == UserFeedback.TEXT
                command_name = "human_feedback"
        else:
            user_input = ""
            # First log new-line so user can differentiate sections better in console
            print()
            if cycles_remaining != math.inf:
                # Print authorized commands left value
                print_attribute(
                    "AUTHORIZED_COMMANDS_LEFT", cycles_remaining, title_color=Fore.CYAN
                )

        ###################
        # Execute Command #
        ###################
        # Decrement the cycle counter first to reduce the likelihood of a SIGINT
        # happening during command execution, setting the cycles remaining to 1,
        # and then having the decrement set it to 0, exiting the application.
        if command_name != "human_feedback":
            cycles_remaining -= 1

        if not command_name:
            continue

        result = await agent.execute(command_name, command_args, user_input)

        if result.status == "success":
            logger.info(result, extra={"title": "SYSTEM:", "title_color": Fore.YELLOW})
        elif result.status == "error":
            logger.warn(
                f"Command {command_name} returned an error: {result.error or result.reason}"
            )


def update_user(
    config: Config,
    ai_config: AIConfig,
    command_name: CommandName,
    command_args: CommandArgs,
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
    logger = logging.getLogger(__name__)

    print_assistant_thoughts(ai_config.ai_name, assistant_reply_dict, config)

    if config.speak_mode:
        speak(f"I want to execute {command_name}")

    # First log new-line so user can differentiate sections better in console
    print()
    logger.info(
        f"COMMAND = {Fore.CYAN}{remove_ansi_escape(command_name)}{Style.RESET_ALL}  "
        f"ARGUMENTS = {Fore.CYAN}{command_args}{Style.RESET_ALL}",
        extra={
            "title": "NEXT ACTION:",
            "title_color": Fore.CYAN,
            "preserve_color": True,
        },
    )


async def get_user_feedback(
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
    logger = logging.getLogger(__name__)

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
            console_input = await clean_input(config, "Waiting for your response...")
        else:
            console_input = await clean_input(
                config, Fore.MAGENTA + "Input: " + Style.RESET_ALL
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


async def construct_main_ai_config(
    config: Config,
    llm_provider: ChatModelProvider,
    name: Optional[str] = None,
    role: Optional[str] = None,
    goals: tuple[str] = tuple(),
) -> AIConfig:
    """Construct the prompt for the AI to respond to

    Returns:
        str: The prompt string
    """
    logger = logging.getLogger(__name__)

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
        print_attribute("Name :", ai_config.ai_name)
        print_attribute("Role :", ai_config.ai_role)
        print_attribute("Goals:", ai_config.ai_goals)
        print_attribute(
            "API Budget:",
            "infinite" if ai_config.api_budget <= 0 else f"${ai_config.api_budget}",
        )
    elif all([ai_config.ai_name, ai_config.ai_role, ai_config.ai_goals]):
        logger.info(
            extra={"title": f"{Fore.GREEN}Welcome back!{Fore.RESET}"},
            msg=f"Would you like me to return to being {ai_config.ai_name}?",
        )
        should_continue = await clean_input(
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
        ai_config = await interactive_ai_config_setup(config, llm_provider)
        ai_config.save(config.workdir / config.ai_settings_file)

    if config.restrict_to_workspace:
        logger.info(
            f"{Fore.YELLOW}NOTE: All files/directories created by this agent"
            f" can be found inside its workspace at:{Fore.RESET} {config.workspace_path}",
            extra={"preserve_color": True},
        )
    # set the total api budget
    api_manager = ApiManager()
    api_manager.set_total_budget(ai_config.api_budget)

    # Agent Created, print message
    logger.info(
        f"{Fore.LIGHTBLUE_EX}{ai_config.ai_name}{Fore.RESET} has been created with the following details:",
        extra={"preserve_color": True},
    )

    # Print the ai_config details
    print_attribute("Name :", ai_config.ai_name)
    print_attribute("Role :", ai_config.ai_role)
    print_attribute("Goals:", "")
    for goal in ai_config.ai_goals:
        logger.info(f"- {goal}")

    return ai_config


def print_assistant_thoughts(
    ai_name: str,
    assistant_reply_json_valid: dict,
    config: Config,
) -> None:
    logger = logging.getLogger(__name__)

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
    print_attribute(
        f"{ai_name.upper()} THOUGHTS", assistant_thoughts_text, title_color=Fore.YELLOW
    )
    print_attribute("REASONING", assistant_thoughts_reasoning, title_color=Fore.YELLOW)
    if assistant_thoughts_plan:
        print_attribute("PLAN", "", title_color=Fore.YELLOW)
        # If it's a list, join it into a string
        if isinstance(assistant_thoughts_plan, list):
            assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
        elif isinstance(assistant_thoughts_plan, dict):
            assistant_thoughts_plan = str(assistant_thoughts_plan)

        # Split the input_string using the newline character and dashes
        lines = assistant_thoughts_plan.split("\n")
        for line in lines:
            line = line.lstrip("- ")
            logger.info(line.strip(), extra={"title": "- ", "title_color": Fore.GREEN})
    print_attribute(
        "CRITICISM", f"{assistant_thoughts_criticism}", title_color=Fore.YELLOW
    )

    # Speak the assistant's thoughts
    if assistant_thoughts_speak:
        if config.speak_mode:
            speak(assistant_thoughts_speak)
        else:
            print_attribute("SPEAK", assistant_thoughts_speak, title_color=Fore.YELLOW)


def remove_ansi_escape(s: str) -> str:
    return s.replace("\x1B", "")
