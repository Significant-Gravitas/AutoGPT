import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style

from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.json_utils.utilities import extract_dict_from_response, validate_dict
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import ChatModelResponse, ChatSequence, Message
from autogpt.llm.utils import count_string_tokens
from autogpt.logs import logger, print_assistant_thoughts, remove_ansi_escape
from autogpt.logs.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.memory.vector import VectorMemory
from autogpt.models.command_registry import CommandRegistry
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace

from .base import BaseAgent


class Agent(BaseAgent):
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_config: AIConfig object encapsulating the agent's personality.
        command_registry: CommandRegistry containing the agent's abilities.
        memory: The memory object to use.
        next_action_count: The number of actions to execute.
        triggering_prompt: The last sentence the AI will see before answering.
            For Auto-GPT, this prompt is:
            Determine exactly one command to use, and respond using the format specified
              above:
            The triggering prompt is not part of the system prompt because between the
              system prompt and the triggering
            prompt we have contextual information that can distract the AI and make it
              forget that its goal is to find the next task to achieve.
            SYSTEM PROMPT
            CONTEXTUAL INFORMATION (memory, previous conversations, anything relevant)
            TRIGGERING PROMPT

            The triggering prompt reminds the AI about its short term meta task
            (defining the next task)
        workspace_directory: Workspace folder that the agent has access to, e.g. for
            reading/writing files.
    """

    def __init__(
        self,
        ai_config: AIConfig,
        command_registry: CommandRegistry,
        memory: VectorMemory,
        next_action_count: int,
        triggering_prompt: str,
        workspace_directory: str | Path,
        config: Config,
    ):
        super().__init__(
            ai_config=ai_config,
            command_registry=command_registry,
            config=config,
            default_cycle_instruction=triggering_prompt,
            cycle_budget=next_action_count,
        )
        self.memory = memory
        self.workspace = Workspace(workspace_directory, config.restrict_to_workspace)
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_cycle_handler = LogCycleHandler()

    def start_interaction_loop(self):
        # Interaction Loop
        self.cycle_count = 0

        # Signal handler for interrupting y -N
        def signal_handler(signum, frame):
            if self.cycle_budget == 0:
                sys.exit()
            else:
                print(
                    Fore.RED
                    + "Interrupt signal received. Stopping continuous command execution."
                    + Style.RESET_ALL
                )
                self.cycle_budget = 0

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            try:
                next(self)
            except StopIteration:
                break

    def think_context(self):
        return Spinner("Thinking... ", plain_output=self.config.plain_output)

    def construct_base_prompt(self, *args, **kwargs) -> ChatSequence:
        if kwargs.get("append_messages") is None:
            kwargs["append_messages"] = []

        # Clock
        kwargs["append_messages"].append(
            Message("system", f"The current time and date is {time.strftime('%c')}"),
        )

        # Add budget information (if any) to prompt
        budget_msg: Message | None = None
        api_manager = ApiManager()
        if api_manager.get_total_budget() > 0.0:
            remaining_budget = (
                api_manager.get_total_budget() - api_manager.get_total_cost()
            )
            if remaining_budget < 0:
                remaining_budget = 0

            budget_msg = Message(
                "system",
                f"Your remaining API budget is ${remaining_budget:.3f}"
                + (
                    " BUDGET EXCEEDED! SHUT DOWN!\n\n"
                    if remaining_budget == 0
                    else " Budget very nearly exceeded! Shut down gracefully!\n\n"
                    if remaining_budget < 0.005
                    else " Budget nearly exceeded. Finish up.\n\n"
                    if remaining_budget < 0.01
                    else ""
                ),
            )
            logger.debug(budget_msg)
            kwargs["append_messages"].append(budget_msg)

        return super().construct_base_prompt(*args, **kwargs)

    def on_before_think(self, *args, **kwargs):
        super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            self.history.raw(),
            FULL_MESSAGE_HISTORY_FILE_NAME,
        )

    def parse_and_process_response(self, llm_response: ChatModelResponse) -> None:
        # Avoid circular imports
        from autogpt.app import execute_command, get_command

        if not llm_response.content:
            raise SyntaxError("Assistant response has no text content")

        assistant_reply_dict = extract_dict_from_response(llm_response.content)

        valid, errors = validate_dict(assistant_reply_dict, self.config)
        if not valid:
            raise SyntaxError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            assistant_reply_dict = plugin.post_planning(assistant_reply_dict)

        command_name: str | None = None
        arguments: dict[str, str] | None = None
        user_input = ""

        # Print Assistant thoughts
        if assistant_reply_dict != {}:
            # Get command name and arguments
            try:
                print_assistant_thoughts(
                    self.ai_config.ai_name, assistant_reply_dict, self.config
                )
                command_name, arguments = get_command(
                    assistant_reply_dict, llm_response, self.config
                )
                if self.config.speak_mode:
                    say_text(f"I want to execute {command_name}", self.config)

            except Exception as e:
                logger.error("Error: \n", str(e))

        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )

        # First log new-line so user can differentiate sections better in console
        logger.typewriter_log("\n")
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{remove_ansi_escape(command_name)}{Style.RESET_ALL}  "
            f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
        )

        if not self.config.continuous_mode and self.next_action_count == 0:
            # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
            # Get key press: Prompt the user to press enter to continue or escape
            # to exit
            self.user_input = ""
            logger.info(
                "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands, "
                "'n' to exit program, or enter feedback for "
                f"{self.ai_config.ai_name}..."
            )
            while True:
                if self.config.chat_messages_enabled:
                    console_input = clean_input(
                        self.config, "Waiting for your response..."
                    )
                else:
                    console_input = clean_input(
                        self.config, Fore.MAGENTA + "Input:" + Style.RESET_ALL
                    )
                if console_input.lower().strip() == self.config.authorise_key:
                    user_input = "GENERATE NEXT COMMAND JSON"
                    break
                elif console_input.lower().strip() == "":
                    logger.warn("Invalid input format.")
                    continue
                elif console_input.lower().startswith(f"{self.config.authorise_key} -"):
                    try:
                        self.next_action_count = abs(int(console_input.split(" ")[1]))
                        user_input = "GENERATE NEXT COMMAND JSON"
                    except ValueError:
                        logger.warn(
                            "Invalid input format. Please enter 'y -n' where n is"
                            " the number of continuous tasks."
                        )
                        continue
                    break
                elif console_input.lower() == self.config.exit_key:
                    user_input = "EXIT"
                    break
                else:
                    user_input = console_input
                    command_name = "human_feedback"
                    self.log_cycle_handler.log_cycle(
                        self.ai_config.ai_name,
                        self.created_at,
                        self.cycle_count,
                        user_input,
                        USER_INPUT_FILE_NAME,
                    )
                    break

            if user_input == "GENERATE NEXT COMMAND JSON":
                logger.typewriter_log(
                    "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                    Fore.MAGENTA,
                    "",
                )
            elif user_input == "EXIT":
                logger.info("Exiting...")
                raise StopIteration
        else:
            # First log new-line so user can differentiate sections better in console
            logger.typewriter_log("\n")
            # Print authorized commands left value
            logger.typewriter_log(
                f"{Fore.CYAN}AUTHORISED COMMANDS LEFT: {Style.RESET_ALL}{self.next_action_count}"
            )

        # Execute command
        if command_name is not None and command_name.lower().startswith("error"):
            result = f"Could not execute command: {arguments}"
        elif command_name == "human_feedback":
            result = f"Human feedback: {user_input}"
        else:
            for plugin in self.config.plugins:
                if not plugin.can_handle_pre_command():
                    continue
                command_name, arguments = plugin.pre_command(command_name, arguments)
            command_result = execute_command(
                command_name=command_name,
                arguments=arguments,
                agent=self,
            )
            result = f"Command {command_name} returned: " f"{command_result}"

            result_tlength = count_string_tokens(
                str(command_result), self.config.fast_llm_model
            )
            memory_tlength = count_string_tokens(
                str(self.history.summary_message()), self.config.fast_llm_model
            )
            if result_tlength + memory_tlength > self.send_token_limit:
                result = f"Failure: command {command_name} returned too much output. \
                    Do not execute this command again with the same arguments."

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                result = plugin.post_command(command_name, result)
            if self.next_action_count > 0:
                self.next_action_count -= 1

        # Check if there's a result from the command append it to the message
        # history
        if result is not None:
            self.history.add("system", result, "action_result")
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
        else:
            self.history.add("system", "Unable to execute command", "action_result")
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")
