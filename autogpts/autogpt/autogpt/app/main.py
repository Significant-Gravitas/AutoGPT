"""
The application entry point. Can be invoked by a CLI or any other front end application.
"""

import enum
import logging
import math
import os
import re
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Optional

from colorama import Fore, Style
from forge.sdk.db import AgentDB

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

from autogpt.agent_factory.configurators import configure_agent_with_state, create_agent
from autogpt.agent_factory.profile_generator import generate_agent_profile_for_task
from autogpt.agent_manager import AgentManager
from autogpt.agents import AgentThoughts, CommandArgs, CommandName
from autogpt.agents.utils.exceptions import AgentTerminated, InvalidAgentResponseError
from autogpt.commands.execute_code import (
    is_docker_available,
    we_are_running_in_a_docker_container,
)
from autogpt.commands.system import finish
from autogpt.config import (
    AIDirectives,
    AIProfile,
    Config,
    ConfigBuilder,
    assert_config_has_openai_api_key,
)
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from autogpt.core.resource.model_providers.llamafile import LlamafileProvider
from autogpt.core.runner.client_lib.utils import coroutine
from autogpt.file_storage import FileStorageBackendName, get_storage
from autogpt.logs.config import configure_chat_plugins, configure_logging
from autogpt.logs.helpers import print_attribute, speak
from autogpt.models.action_history import ActionInterruptedByHuman
from autogpt.plugins import scan_plugins
from scripts.install_plugin_deps import install_plugin_dependencies

from .configurator import apply_overrides_to_config
from .setup import apply_overrides_to_ai_settings, interactively_revise_ai_settings
from .spinner import Spinner
from .utils import (
    clean_input,
    get_legal_warning,
    markdown_to_ansi_style,
    print_git_branch_info,
    print_motd,
    print_python_version_info,
)


@coroutine
async def run_auto_gpt(
    continuous: bool = False,
    continuous_limit: Optional[int] = None,
    ai_settings: Optional[Path] = None,
    prompt_settings: Optional[Path] = None,
    skip_reprompt: bool = False,
    speak: bool = False,
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    gpt3only: bool = False,
    gpt4only: bool = False,
    browser_name: Optional[str] = None,
    allow_downloads: bool = False,
    skip_news: bool = False,
    workspace_directory: Optional[Path] = None,
    install_plugin_deps: bool = False,
    override_ai_name: Optional[str] = None,
    override_ai_role: Optional[str] = None,
    resources: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    best_practices: Optional[list[str]] = None,
    override_directives: bool = False,
):
    # Set up configuration
    config = ConfigBuilder.build_config_from_env()
    # Storage
    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend, root_path="data", restrict_to_root=restrict_to_root
    )
    file_storage.initialize()

    # Set up logging module
    configure_logging(
        **config.logging.dict(),
        tts_config=config.tts_config,
    )

    # TODO: fill in llm values here
    assert_config_has_openai_api_key(config)

    apply_overrides_to_config(
        config=config,
        continuous=continuous,
        continuous_limit=continuous_limit,
        ai_settings_file=ai_settings,
        prompt_settings_file=prompt_settings,
        skip_reprompt=skip_reprompt,
        speak=speak,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
        skip_news=skip_news,
    )

    llm_provider = _configure_openai_provider(config)

    logger = logging.getLogger(__name__)

    if config.continuous_mode:
        for line in get_legal_warning().split("\n"):
            logger.warning(
                extra={
                    "title": "LEGAL:",
                    "title_color": Fore.RED,
                    "preserve_color": True,
                },
                msg=markdown_to_ansi_style(line),
            )

    if not config.skip_news:
        print_motd(config, logger)
        print_git_branch_info(logger)
        print_python_version_info(logger)
        print_attribute("Smart LLM", config.smart_llm)
        print_attribute("Fast LLM", config.fast_llm)
        print_attribute("Browser", config.selenium_web_browser)
        if config.continuous_mode:
            print_attribute("Continuous Mode", "ENABLED", title_color=Fore.YELLOW)
            if continuous_limit:
                print_attribute("Continuous Limit", config.continuous_limit)
        if config.tts_config.speak_mode:
            print_attribute("Speak Mode", "ENABLED")
        if ai_settings:
            print_attribute("Using AI Settings File", ai_settings)
        if prompt_settings:
            print_attribute("Using Prompt Settings File", prompt_settings)
        if config.allow_downloads:
            print_attribute("Native Downloading", "ENABLED")
        if we_are_running_in_a_docker_container() or is_docker_available():
            print_attribute("Code Execution", "ENABLED")
        else:
            print_attribute(
                "Code Execution",
                "DISABLED (Docker unavailable)",
                title_color=Fore.YELLOW,
            )

    if install_plugin_deps:
        install_plugin_dependencies()

    config.plugins = scan_plugins(config)
    configure_chat_plugins(config)

    # Let user choose an existing agent to run
    agent_manager = AgentManager(file_storage)
    existing_agents = agent_manager.list_agents()
    load_existing_agent = ""
    if existing_agents:
        print(
            "Existing agents\n---------------\n"
            + "\n".join(f"{i} - {id}" for i, id in enumerate(existing_agents, 1))
        )
        load_existing_agent = clean_input(
            config,
            "Enter the number or name of the agent to run,"
            " or hit enter to create a new one:",
        )
        if re.match(r"^\d+$", load_existing_agent.strip()) and 0 < int(
            load_existing_agent
        ) <= len(existing_agents):
            load_existing_agent = existing_agents[int(load_existing_agent) - 1]

        if load_existing_agent not in existing_agents:
            logger.info(
                f"Unknown agent '{load_existing_agent}', "
                f"creating a new one instead.",
                extra={"color": Fore.YELLOW},
            )
            load_existing_agent = ""

    # Either load existing or set up new agent state
    agent = None
    agent_state = None

    ############################
    # Resume an Existing Agent #
    ############################
    if load_existing_agent:
        agent_state = None
        while True:
            answer = clean_input(config, "Resume? [Y/n]")
            if answer == "" or answer.lower() == "y":
                agent_state = agent_manager.load_agent_state(load_existing_agent)
                break
            elif answer.lower() == "n":
                break

    if agent_state:
        agent = configure_agent_with_state(
            state=agent_state,
            app_config=config,
            file_storage=file_storage,
            llm_provider=llm_provider,
        )
        apply_overrides_to_ai_settings(
            ai_profile=agent.state.ai_profile,
            directives=agent.state.directives,
            override_name=override_ai_name,
            override_role=override_ai_role,
            resources=resources,
            constraints=constraints,
            best_practices=best_practices,
            replace_directives=override_directives,
        )

        if (
            agent.event_history.current_episode
            and agent.event_history.current_episode.action.name == finish.__name__
            and not agent.event_history.current_episode.result
        ):
            # Agent was resumed after `finish` -> rewrite result of `finish` action
            finish_reason = agent.event_history.current_episode.action.args["reason"]
            print(f"Agent previously self-terminated; reason: '{finish_reason}'")
            new_assignment = clean_input(
                config, "Please give a follow-up question or assignment:"
            )
            agent.event_history.register_result(
                ActionInterruptedByHuman(feedback=new_assignment)
            )

        # If any of these are specified as arguments,
        #  assume the user doesn't want to revise them
        if not any(
            [
                override_ai_name,
                override_ai_role,
                resources,
                constraints,
                best_practices,
            ]
        ):
            ai_profile, ai_directives = await interactively_revise_ai_settings(
                ai_profile=agent.state.ai_profile,
                directives=agent.state.directives,
                app_config=config,
            )
        else:
            logger.info("AI config overrides specified through CLI; skipping revision")

    ######################
    # Set up a new Agent #
    ######################
    if not agent:
        task = ""
        while task.strip() == "":
            task = clean_input(
                config,
                "Enter the task that you want AutoGPT to execute,"
                " with as much detail as possible:",
            )

        base_ai_directives = AIDirectives.from_file(config.prompt_settings_file)

        ai_profile, task_oriented_ai_directives = await generate_agent_profile_for_task(
            task,
            app_config=config,
            llm_provider=llm_provider,
        )
        ai_directives = base_ai_directives + task_oriented_ai_directives
        apply_overrides_to_ai_settings(
            ai_profile=ai_profile,
            directives=ai_directives,
            override_name=override_ai_name,
            override_role=override_ai_role,
            resources=resources,
            constraints=constraints,
            best_practices=best_practices,
            replace_directives=override_directives,
        )

        # If any of these are specified as arguments,
        #  assume the user doesn't want to revise them
        if not any(
            [
                override_ai_name,
                override_ai_role,
                resources,
                constraints,
                best_practices,
            ]
        ):
            ai_profile, ai_directives = await interactively_revise_ai_settings(
                ai_profile=ai_profile,
                directives=ai_directives,
                app_config=config,
            )
        else:
            logger.info("AI config overrides specified through CLI; skipping revision")

        agent = create_agent(
            agent_id=agent_manager.generate_id(ai_profile.ai_name),
            task=task,
            ai_profile=ai_profile,
            directives=ai_directives,
            app_config=config,
            file_storage=file_storage,
            llm_provider=llm_provider,
        )

        if not agent.config.allow_fs_access:
            logger.info(
                f"{Fore.YELLOW}"
                "NOTE: All files/directories created by this agent can be found "
                f"inside its workspace at:{Fore.RESET} {agent.workspace.root}",
                extra={"preserve_color": True},
            )

    #################
    # Run the Agent #
    #################
    try:
        await run_interaction_loop(agent)
    except AgentTerminated:
        agent_id = agent.state.agent_id
        logger.info(f"Saving state of {agent_id}...")

        # Allow user to Save As other ID
        save_as_id = clean_input(
            config,
            f"Press enter to save as '{agent_id}',"
            " or enter a different ID to save to:",
        )
        # TODO: allow many-to-one relations of agents and workspaces
        await agent.save_state(save_as_id if not save_as_id.isspace() else None)


@coroutine
async def run_auto_gpt_server(
    prompt_settings: Optional[Path] = None,
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    gpt3only: bool = False,
    gpt4only: bool = False,
    browser_name: Optional[str] = None,
    allow_downloads: bool = False,
    install_plugin_deps: bool = False,
):
    from .agent_protocol_server import AgentProtocolServer

    config = ConfigBuilder.build_config_from_env()
    # Storage
    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend, root_path="data", restrict_to_root=restrict_to_root
    )
    file_storage.initialize()

    # Set up logging module
    configure_logging(
        **config.logging.dict(),
        tts_config=config.tts_config,
    )

    # TODO: fill in llm values here
    assert_config_has_openai_api_key(config)

    apply_overrides_to_config(
        config=config,
        prompt_settings_file=prompt_settings,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        gpt3only=gpt3only,
        gpt4only=gpt4only,
        browser_name=browser_name,
        allow_downloads=allow_downloads,
    )

    llm_provider = _configure_openai_provider(config)

    if install_plugin_deps:
        install_plugin_dependencies()

    config.plugins = scan_plugins(config)

    # Set up & start server
    database = AgentDB(
        database_string=os.getenv("AP_SERVER_DB_URL", "sqlite:///data/ap_server.db"),
        debug_enabled=debug,
    )
    port: int = int(os.getenv("AP_SERVER_PORT", default=8000))
    server = AgentProtocolServer(
        app_config=config,
        database=database,
        file_storage=file_storage,
        llm_provider=llm_provider,
    )
    await server.start(port=port)

    logging.getLogger().info(
        f"Total OpenAI session cost: "
        f"${round(sum(b.total_cost for b in server._task_budgets.values()), 2)}"
    )


def _configure_openai_provider(config: Config) -> OpenAIProvider:
    """Create a configured OpenAIProvider object.

    Args:
        config: The program's configuration.

    Returns:
        A configured OpenAIProvider object.
    """
    if config.openai_credentials is None:
        raise RuntimeError("OpenAI key is not configured")

    openai_settings = OpenAIProvider.default_settings.copy(deep=True)
    openai_settings.credentials = config.openai_credentials

    if config.llm_provider == "llamafile":
        return LlamafileProvider(
            settings=openai_settings,
            logger=logging.getLogger("LlamafileProvider"),
        )
    else:
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
    agent: "Agent",
) -> None:
    """Run the main interaction loop for the agent.

    Args:
        agent: The agent to run the interaction loop for.

    Returns:
        None
    """
    # These contain both application config and agent config, so grab them here.
    legacy_config = agent.legacy_config
    ai_profile = agent.ai_profile
    logger = logging.getLogger(__name__)

    cycle_budget = cycles_remaining = _get_cycle_budget(
        legacy_config.continuous_mode, legacy_config.continuous_limit
    )
    spinner = Spinner(
        "Thinking...", plain_output=legacy_config.logging.plain_console_output
    )
    stop_reason = None

    def graceful_agent_interrupt(signum: int, frame: Optional[FrameType]) -> None:
        nonlocal cycle_budget, cycles_remaining, spinner, stop_reason
        if stop_reason:
            logger.error("Quitting immediately...")
            sys.exit()
        if cycles_remaining in [0, 1]:
            logger.warning("Interrupt signal received: shutting down gracefully.")
            logger.warning(
                "Press Ctrl+C again if you want to stop AutoGPT immediately."
            )
            stop_reason = AgentTerminated("Interrupt signal received")
        else:
            restart_spinner = spinner.running
            if spinner.running:
                spinner.stop()

            logger.error(
                "Interrupt signal received: stopping continuous command execution."
            )
            cycles_remaining = 1
            if restart_spinner:
                spinner.start()

    def handle_stop_signal() -> None:
        if stop_reason:
            raise stop_reason

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
        handle_stop_signal()
        # Have the agent determine the next action to take.
        with spinner:
            try:
                (
                    command_name,
                    command_args,
                    assistant_reply_dict,
                ) = await agent.propose_action()
            except InvalidAgentResponseError as e:
                logger.warning(f"The agent's thoughts could not be parsed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.error(
                        "The agent failed to output valid thoughts"
                        f" {consecutive_failures} times in a row. Terminating..."
                    )
                    raise AgentTerminated(
                        "The agent failed to output valid thoughts"
                        f" {consecutive_failures} times in a row."
                    )
                continue

        consecutive_failures = 0

        ###############
        # Update User #
        ###############
        # Print the assistant's thoughts and the next command to the user.
        update_user(
            ai_profile,
            command_name,
            command_args,
            assistant_reply_dict,
            speak_mode=legacy_config.tts_config.speak_mode,
        )

        ##################
        # Get user input #
        ##################
        handle_stop_signal()
        if cycles_remaining == 1:  # Last cycle
            user_feedback, user_input, new_cycles_remaining = await get_user_feedback(
                legacy_config,
                ai_profile,
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
                logger.warning("Exiting...")
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

        handle_stop_signal()

        if command_name:
            result = await agent.execute(command_name, command_args, user_input)

            if result.status == "success":
                logger.info(
                    result, extra={"title": "SYSTEM:", "title_color": Fore.YELLOW}
                )
            elif result.status == "error":
                logger.warning(
                    f"Command {command_name} returned an error: "
                    f"{result.error or result.reason}"
                )


def update_user(
    ai_profile: AIProfile,
    command_name: CommandName,
    command_args: CommandArgs,
    assistant_reply_dict: AgentThoughts,
    speak_mode: bool = False,
) -> None:
    """Prints the assistant's thoughts and the next command to the user.

    Args:
        config: The program's configuration.
        ai_profile: The AI's personality/profile
        command_name: The name of the command to execute.
        command_args: The arguments for the command.
        assistant_reply_dict: The assistant's reply.
    """
    logger = logging.getLogger(__name__)

    print_assistant_thoughts(
        ai_name=ai_profile.ai_name,
        assistant_reply_json_valid=assistant_reply_dict,
        speak_mode=speak_mode,
    )

    if speak_mode:
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
    ai_profile: AIProfile,
) -> tuple[UserFeedback, str, int | None]:
    """Gets the user's feedback on the assistant's reply.

    Args:
        config: The program's configuration.
        ai_profile: The AI's configuration.

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
        f"{ai_profile.ai_name}..."
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
            logger.warning("Invalid input format.")
        elif console_input.lower().startswith(f"{config.authorise_key} -"):
            try:
                user_feedback = UserFeedback.AUTHORIZE
                new_cycles_remaining = abs(int(console_input.split(" ")[1]))
            except ValueError:
                logger.warning(
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


def print_assistant_thoughts(
    ai_name: str,
    assistant_reply_json_valid: dict,
    speak_mode: bool = False,
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
            assistant_thoughts.get("self_criticism", "")
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
        if speak_mode:
            speak(assistant_thoughts_speak)
        else:
            print_attribute("SPEAK", assistant_thoughts_speak, title_color=Fore.YELLOW)


def remove_ansi_escape(s: str) -> str:
    return s.replace("\x1B", "")
