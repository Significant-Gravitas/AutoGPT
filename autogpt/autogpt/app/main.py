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
from forge.agent_protocol.database import AgentDB
from forge.components.code_executor.code_executor import (
    is_docker_available,
    we_are_running_in_a_docker_container,
)
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.file_storage import FileStorageBackendName, get_storage
from forge.llm.providers import MultiProvider
from forge.logging.config import configure_logging
from forge.logging.utils import print_attribute, speak
from forge.models.action import ActionInterruptedByHuman, ActionProposal
from forge.models.utils import ModelWithSummary
from forge.utils.const import FINISH_COMMAND
from forge.utils.exceptions import AgentTerminated, InvalidAgentResponseError

from autogpt.agent_factory.configurators import configure_agent_with_state, create_agent
from autogpt.agents.agent_manager import AgentManager
from autogpt.agents.prompt_strategies.one_shot import AssistantThoughts
from autogpt.app.config import (
    AppConfig,
    ConfigBuilder,
    assert_config_has_required_llm_api_keys,
)

if TYPE_CHECKING:
    from autogpt.agents.agent import Agent

from .configurator import apply_overrides_to_config
from .input import clean_input
from .setup import apply_overrides_to_ai_settings, interactively_revise_ai_settings
from .spinner import Spinner
from .utils import (
    coroutine,
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
    skip_reprompt: bool = False,
    speak: bool = False,
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    skip_news: bool = False,
    install_plugin_deps: bool = False,
    override_ai_name: Optional[str] = None,
    override_ai_role: Optional[str] = None,
    resources: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    best_practices: Optional[list[str]] = None,
    override_directives: bool = False,
    component_config_file: Optional[Path] = None,
):
    # Set up configuration
    config = ConfigBuilder.build_config_from_env()
    # Storage
    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend,
        root_path=Path("data"),
        restrict_to_root=restrict_to_root,
    )
    file_storage.initialize()

    # Set up logging module
    if speak:
        config.tts_config.speak_mode = True
    configure_logging(
        debug=debug,
        level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        config=config.logging,
        tts_config=config.tts_config,
    )

    await assert_config_has_required_llm_api_keys(config)

    await apply_overrides_to_config(
        config=config,
        continuous=continuous,
        continuous_limit=continuous_limit,
        skip_reprompt=skip_reprompt,
        skip_news=skip_news,
    )

    llm_provider = _configure_llm_provider(config)

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
        print_motd(logger)
        print_git_branch_info(logger)
        print_python_version_info(logger)
        print_attribute("Smart LLM", config.smart_llm)
        print_attribute("Fast LLM", config.fast_llm)
        if config.continuous_mode:
            print_attribute("Continuous Mode", "ENABLED", title_color=Fore.YELLOW)
            if continuous_limit:
                print_attribute("Continuous Limit", config.continuous_limit)
        if config.tts_config.speak_mode:
            print_attribute("Speak Mode", "ENABLED")
        if we_are_running_in_a_docker_container() or is_docker_available():
            print_attribute("Code Execution", "ENABLED")
        else:
            print_attribute(
                "Code Execution",
                "DISABLED (Docker unavailable)",
                title_color=Fore.YELLOW,
            )

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
            "Enter the number or name of the agent to run,"
            " or hit enter to create a new one:",
        )
        if re.match(r"^\d+$", load_existing_agent.strip()) and 0 < int(
            load_existing_agent
        ) <= len(existing_agents):
            load_existing_agent = existing_agents[int(load_existing_agent) - 1]

        if load_existing_agent != "" and load_existing_agent not in existing_agents:
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
            answer = clean_input("Resume? [Y/n]")
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
            (current_episode := agent.event_history.current_episode)
            and current_episode.action.use_tool.name == FINISH_COMMAND
            and not current_episode.result
        ):
            # Agent was resumed after `finish` -> rewrite result of `finish` action
            finish_reason = current_episode.action.use_tool.arguments["reason"]
            print(f"Agent previously self-terminated; reason: '{finish_reason}'")
            new_assignment = clean_input(
                "Please give a follow-up question or assignment:"
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
                "Enter the task that you want AutoGPT to execute,"
                " with as much detail as possible:",
            )

        ai_profile = AIProfile()
        additional_ai_directives = AIDirectives()
        apply_overrides_to_ai_settings(
            ai_profile=ai_profile,
            directives=additional_ai_directives,
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
            (
                ai_profile,
                additional_ai_directives,
            ) = await interactively_revise_ai_settings(
                ai_profile=ai_profile,
                directives=additional_ai_directives,
                app_config=config,
            )
        else:
            logger.info("AI config overrides specified through CLI; skipping revision")

        agent = create_agent(
            agent_id=agent_manager.generate_id(ai_profile.ai_name),
            task=task,
            ai_profile=ai_profile,
            directives=additional_ai_directives,
            app_config=config,
            file_storage=file_storage,
            llm_provider=llm_provider,
        )

        file_manager = agent.file_manager

        if file_manager and not agent.config.allow_fs_access:
            logger.info(
                f"{Fore.YELLOW}"
                "NOTE: All files/directories created by this agent can be found "
                f"inside its workspace at:{Fore.RESET} {file_manager.workspace.root}",
                extra={"preserve_color": True},
            )

        # TODO: re-evaluate performance benefit of task-oriented profiles
        # # Concurrently generate a custom profile for the agent and apply it once done
        # def update_agent_directives(
        #     task: asyncio.Task[tuple[AIProfile, AIDirectives]]
        # ):
        #     logger.debug(f"Updating AIProfile: {task.result()[0]}")
        #     logger.debug(f"Adding AIDirectives: {task.result()[1]}")
        #     agent.state.ai_profile = task.result()[0]
        #     agent.state.directives = agent.state.directives + task.result()[1]

        # asyncio.create_task(
        #     generate_agent_profile_for_task(
        #         task, app_config=config, llm_provider=llm_provider
        #     )
        # ).add_done_callback(update_agent_directives)

    # Load component configuration from file
    if _config_file := component_config_file or config.component_config_file:
        try:
            logger.info(f"Loading component configuration from {_config_file}")
            agent.load_component_configs(_config_file.read_text())
        except Exception as e:
            logger.error(f"Could not load component configuration: {e}")

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
            f"Press enter to save as '{agent_id}',"
            " or enter a different ID to save to:",
        )
        # TODO: allow many-to-one relations of agents and workspaces
        await agent.file_manager.save_state(
            save_as_id.strip() if not save_as_id.isspace() else None
        )


@coroutine
async def run_auto_gpt_server(
    debug: bool = False,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file_format: Optional[str] = None,
    install_plugin_deps: bool = False,
):
    from .agent_protocol_server import AgentProtocolServer

    config = ConfigBuilder.build_config_from_env()
    # Storage
    local = config.file_storage_backend == FileStorageBackendName.LOCAL
    restrict_to_root = not local or config.restrict_to_workspace
    file_storage = get_storage(
        config.file_storage_backend,
        root_path=Path("data"),
        restrict_to_root=restrict_to_root,
    )
    file_storage.initialize()

    # Set up logging module
    configure_logging(
        debug=debug,
        level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        config=config.logging,
        tts_config=config.tts_config,
    )

    await assert_config_has_required_llm_api_keys(config)

    await apply_overrides_to_config(
        config=config,
    )

    llm_provider = _configure_llm_provider(config)

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


def _configure_llm_provider(config: AppConfig) -> MultiProvider:
    multi_provider = MultiProvider()
    for model in [config.smart_llm, config.fast_llm]:
        # Ensure model providers for configured LLMs are available
        multi_provider.get_model_provider(model)
    return multi_provider


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
    app_config = agent.app_config
    ai_profile = agent.state.ai_profile
    logger = logging.getLogger(__name__)

    cycle_budget = cycles_remaining = _get_cycle_budget(
        app_config.continuous_mode, app_config.continuous_limit
    )
    spinner = Spinner(
        "Thinking...", plain_output=app_config.logging.plain_console_output
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
        if not (_ep := agent.event_history.current_episode) or _ep.result:
            with spinner:
                try:
                    action_proposal = await agent.propose_action()
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
        else:
            action_proposal = _ep.action

        consecutive_failures = 0

        ###############
        # Update User #
        ###############
        # Print the assistant's thoughts and the next command to the user.
        update_user(
            ai_profile,
            action_proposal,
            speak_mode=app_config.tts_config.speak_mode,
        )

        ##################
        # Get user input #
        ##################
        handle_stop_signal()
        if cycles_remaining == 1:  # Last cycle
            feedback_type, feedback, new_cycles_remaining = await get_user_feedback(
                app_config,
                ai_profile,
            )

            if feedback_type == UserFeedback.AUTHORIZE:
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
            elif feedback_type == UserFeedback.EXIT:
                logger.warning("Exiting...")
                exit()
            else:  # user_feedback == UserFeedback.TEXT
                pass
        else:
            feedback = ""
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
        if not feedback:
            cycles_remaining -= 1

        if not action_proposal.use_tool:
            continue

        handle_stop_signal()

        if not feedback:
            result = await agent.execute(action_proposal)
        else:
            result = await agent.do_not_execute(action_proposal, feedback)

        if result.status == "success":
            logger.info(result, extra={"title": "SYSTEM:", "title_color": Fore.YELLOW})
        elif result.status == "error":
            logger.warning(
                f"Command {action_proposal.use_tool.name} returned an error: "
                f"{result.error or result.reason}"
            )


def update_user(
    ai_profile: AIProfile,
    action_proposal: "ActionProposal",
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
        thoughts=action_proposal.thoughts,
        speak_mode=speak_mode,
    )

    if speak_mode:
        speak(f"I want to execute {action_proposal.use_tool.name}")

    # First log new-line so user can differentiate sections better in console
    print()
    safe_tool_name = remove_ansi_escape(action_proposal.use_tool.name)
    logger.info(
        f"COMMAND = {Fore.CYAN}{safe_tool_name}{Style.RESET_ALL}  "
        f"ARGUMENTS = {Fore.CYAN}{action_proposal.use_tool.arguments}{Style.RESET_ALL}",
        extra={
            "title": "NEXT ACTION:",
            "title_color": Fore.CYAN,
            "preserve_color": True,
        },
    )


async def get_user_feedback(
    config: AppConfig,
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
        console_input = clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)

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
    thoughts: str | ModelWithSummary | AssistantThoughts,
    speak_mode: bool = False,
) -> None:
    logger = logging.getLogger(__name__)

    thoughts_text = remove_ansi_escape(
        thoughts.text
        if isinstance(thoughts, AssistantThoughts)
        else thoughts.summary()
        if isinstance(thoughts, ModelWithSummary)
        else thoughts
    )
    print_attribute(
        f"{ai_name.upper()} THOUGHTS", thoughts_text, title_color=Fore.YELLOW
    )

    if isinstance(thoughts, AssistantThoughts):
        print_attribute(
            "REASONING", remove_ansi_escape(thoughts.reasoning), title_color=Fore.YELLOW
        )
        if assistant_thoughts_plan := remove_ansi_escape(
            "\n".join(f"- {p}" for p in thoughts.plan)
        ):
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
                logger.info(
                    line.strip(), extra={"title": "- ", "title_color": Fore.GREEN}
                )
        print_attribute(
            "CRITICISM",
            remove_ansi_escape(thoughts.self_criticism),
            title_color=Fore.YELLOW,
        )

        # Speak the assistant's thoughts
        if assistant_thoughts_speak := remove_ansi_escape(thoughts.speak):
            if speak_mode:
                speak(assistant_thoughts_speak)
            else:
                print_attribute(
                    "SPEAK", assistant_thoughts_speak, title_color=Fore.YELLOW
                )
    else:
        speak(thoughts_text)


def remove_ansi_escape(s: str) -> str:
    return s.replace("\x1B", "")
