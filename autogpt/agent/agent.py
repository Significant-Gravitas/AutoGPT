import asyncio
import concurrent.futures
import json
import signal
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from colorama import Fore, Style

from autogpt.commands.loopwatcher import LoopWatcher
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.json_utils.utilities import extract_json_from_response, validate_json
from autogpt.llm.base import ChatModelResponse
from autogpt.llm.chat import chat_with_ai
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS
from autogpt.llm.utils import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts, remove_ansi_escape
from autogpt.memory.message_history import MessageHistory
from autogpt.memory.vector import VectorMemory
from autogpt.models.command import CommandInstance
from autogpt.models.command_registry import CommandRegistry, AgentCommandRegistry
from autogpt.speech import say_text
from autogpt.spinner import Spinner, SpinnerInterrupted
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


class InteractionResult(Enum):
    OK = 0
    SoftInterrupt = 1
    HardInterrupt = 2
    ExceptionInValidation = 3


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        next_action_count: The number of actions to execute.
        system_prompt: The system prompt is the initial prompt that defines everything
          the AI needs to know to achieve its task successfully.
        Currently, the dynamic and customizable information in the system prompt are
          ai_name, description and goals.

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
    """

    def __init__(
        self,
        ai_name: str,
        memory: VectorMemory,
        next_action_count: int,
        command_registry: CommandRegistry,
        ai_config: AIConfig,
        system_prompt: str,
        triggering_prompt: str,
        workspace_directory: str | Path,
        config: Config,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.history = MessageHistory(self)
        self.next_action_count = next_action_count
        self.command_registry = AgentCommandRegistry(self, command_registry)
        self.config = config
        self.ai_config = ai_config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, config.restrict_to_workspace)
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cycle_count = 0
        self.log_cycle_handler = LogCycleHandler()
        self.fast_token_limit = OPEN_AI_CHAT_MODELS.get(
            config.fast_llm_model
        ).max_tokens
        self.loopwatcher = LoopWatcher()

    @staticmethod
    async def async_task_and_spin(
        spn: Spinner, some_task: Callable, args: Tuple
    ) -> Optional[Any]:
        loop = asyncio.get_event_loop()
        # Run the synchronous function in an executor
        pool = concurrent.futures.ThreadPoolExecutor()
        try:
            task = loop.run_in_executor(pool, some_task, *args)
            event_task = loop.run_in_executor(pool, spn.ended.wait)
            # Wait for the task or the event to complete
            done, pending = await asyncio.wait(
                {task, event_task}, return_when=asyncio.FIRST_COMPLETED
            )
            # Cancel any pending tasks
            for t in pending:
                t.cancel()
            # Check which task completed
            if task in done:
                result = await task
                return result
            return None

        finally:
            pool.shutdown(wait=False)  # dont want to use 'with' because it waits...

    def get_command_instance(
        self, assistant_reply_json: Dict, assistant_reply: ChatModelResponse
    ) -> CommandInstance:
        """Parse the response and return the command name and arguments

        Args:
            assistant_reply_json (dict): The response object from the AI
            assistant_reply (ChatModelResponse): The model response from the AI
            config (Config): The config object

        Returns:
            tuple: The command name and arguments

        Raises:
            json.decoder.JSONDecodeError: If the response is not valid JSON

            Exception: If any other error occurs
        """
        if self.config.openai_functions:
            if assistant_reply.function_call is None:
                return "Error:", "No 'function_call' in assistant reply"
            assistant_reply_json["command"] = {
                "name": assistant_reply.function_call.name,
                "args": json.loads(assistant_reply.function_call.arguments),
            }
        try:
            if "command" not in assistant_reply_json:
                return "Error:", "Missing 'command' object in JSON"

            if not isinstance(assistant_reply_json, dict):
                return (
                    "Error:",
                    f"The previous message sent was not a dictionary {assistant_reply_json}",
                )

            command = assistant_reply_json["command"]
            if not isinstance(command, dict):
                return "Error:", "'command' object is not a dictionary"

            if "name" not in command:
                return "Error:", "Missing 'name' field in 'command' object"

            command_name = command["name"]

            # Use an empty dictionary if 'args' field is not present in 'command' object
            arguments = command.get("args", {})

            return self.command_registry.get_command(command_name).generate_instance(
                arguments, agent=self
            )
        except json.decoder.JSONDecodeError:
            return "Error:", "Invalid JSON"
        # All other errors, return "Error: + error message"
        except Exception as e:
            return "Error:", str(e)

    def start_interaction_loop(self) -> None:
        def enter_input() -> None:
            logger.info(
                "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands, "
                "'n' to exit program, or enter feedback for "
                f"{self.ai_name}..."
            )

        def get_user_choice() -> bool:  # False to continue loop , True to break
            nonlocal console_input, user_input, command_name  # type: ignore

            if console_input.lower() == self.config.exit_key:
                user_input = "EXIT"
                return True
            elif console_input.lower().strip() == "":
                logger.warn("Invalid input format.")
                return False

            if console_input.lower().strip() == self.config.authorise_key:
                user_input = "GENERATE NEXT COMMAND JSON"
                return True
            elif console_input.lower().startswith(f"{self.config.authorise_key} -"):
                try:
                    self.next_action_count = abs(int(console_input.split(" ")[1]))
                    user_input = "GENERATE NEXT COMMAND JSON"
                except ValueError:
                    logger.warn(
                        "Invalid input format. Please enter 'y -n' where n is"
                        " the number of continuous tasks."
                    )
                    return False
                return True

            user_input = console_input
            command_name = "human_feedback"
            self.log_cycle_handler.log_cycle(
                self.ai_config.ai_name,
                self.created_at,
                self.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )
            return True

        def print_next_command(is_next: bool) -> None:
            self.log_cycle_handler.log_cycle(
                self.ai_config.ai_name,
                self.created_at,
                self.cycle_count,
                assistant_reply_json,
                NEXT_ACTION_FILE_NAME,
            )

            # First log new-line so user can differentiate sections better in console
            logger.typewriter_log("\n")
            logger.typewriter_log(
                "NEXT ACTION: " if is_next else "LAST ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{remove_ansi_escape(command_name)}{Style.RESET_ALL}  "
                f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
            )

        # Interaction Loop
        self.cycle_count = 0

        arguments = None
        user_input = ""

        # Signal handler for interrupting y -N
        def signal_handler(signum: int, frame) -> None:  # type: ignore
            if self.next_action_count == 0:
                sys.exit()
            else:
                print(
                    Fore.RED
                    + "Interrupt signal received. Stopping continuous command execution."
                    + Style.RESET_ALL
                )
                self.next_action_count = 0

        signal.signal(signal.SIGINT, signal_handler)
        command_name = ""

        while True:
            # Discontinue if continuous limit is reached
            self.cycle_count += 1
            self.log_cycle_handler.log_count_within_cycle = 0
            self.log_cycle_handler.log_cycle(
                self.ai_config.ai_name,
                self.created_at,
                self.cycle_count,
                [m.raw() for m in self.history],
                FULL_MESSAGE_HISTORY_FILE_NAME,
            )
            if (
                self.config.continuous_mode
                and self.config.continuous_limit > 0
                and self.cycle_count > self.config.continuous_limit
            ):
                logger.typewriter_log(
                    "Continuous Limit Reached: ",
                    Fore.YELLOW,
                    f"{self.config.continuous_limit}",
                )
                break

            (
                status,
                assistant_reply,
                assistant_reply_json,
            ) = self.interact_with_assistant()

            tostop = True
            # Print Assistant thoughts
            if assistant_reply_json != {}:
                # Get command name and arguments
                cmd = self.get_next_command_to_execute(
                    assistant_reply, assistant_reply_json
                )
                tostop = self.loopwatcher.should_stop_on_command(cmd)

            tostop = tostop or (
                status != InteractionResult.OK
            )  # stop also in exception

            if command_name != "":
                print_next_command(
                    (status != InteractionResult.HardInterrupt)
                    and (status != InteractionResult.ExceptionInValidation)
                )
            else:
                tostop = True

            if command_name in self.config.commands_to_ignore:
                user_input = "GENERATE NEXT COMMAND JSON"
            elif (
                (not self.config.continuous_mode and self.next_action_count == 0)
                or tostop
                or command_name in self.config.commands_to_stop
            ):
                # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.info(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous commands "
                    "'n' to exit program, or enter feedback for "
                    f"{self.ai_name}..."
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
                    if get_user_choice():
                        break

                if user_input == "GENERATE NEXT COMMAND JSON":
                    self.loopwatcher.command_authorized(hash(cmd))
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )
                elif user_input == "EXIT":
                    logger.info("Exiting...")
                    break

            else:
                # First log new-line so user can differentiate sections better in console
                logger.typewriter_log("\n")
                # Print authorized commands left value
                logger.typewriter_log(
                    f"{Fore.CYAN}AUTHORISED COMMANDS LEFT: {Style.RESET_ALL}{self.next_action_count}"
                )

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                result = (
                    f"Command {command_name} threw the following error: {arguments}"
                )
            elif command_name == "human_feedback":
                result = f"Human feedback: {user_input}"
            else:
                for plugin in self.config.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(
                        command_name, arguments
                    )
                command_result = cmd.execute()

                result = f"Command {command_name} returned: " f"{command_result}"

                result_tlength = count_string_tokens(
                    str(command_result), self.config.fast_llm_model
                )
                memory_tlength = count_string_tokens(
                    str(self.history.summary_message()), self.config.fast_llm_model
                )
                if result_tlength + memory_tlength + 600 > self.fast_token_limit:
                    result = f"Failure: command {command_name} returned too much output. \
                        Do not execute this command again with the same arguments."

                for plugin in self.config.plugins:
                    if not plugin.can_handle_post_command():
                        continue
                    result = plugin.post_command(command_name, result)
                if (
                    self.next_action_count > 0
                    and command_name not in self.config.commands_to_ignore
                ):
                    self.next_action_count -= 1

            # Check if there's a result from the command append it to the message
            # history
            if result is not None:
                self.history.add("system", result, "action_result")
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                self.history.add("system", "Unable to execute command", "action_result")
                logger.typewriter_log(
                    "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                )

    def get_next_command_to_execute(
        self, assistant_reply: ChatModelResponse, assistant_reply_json: Dict
    ) -> CommandInstance:
        command_name, arguments = None, None
        try:
            print_assistant_thoughts(self.ai_name, assistant_reply_json, self.config)

            cmd = self.get_command_instance(assistant_reply, assistant_reply_json)

            assistant_reply_json, assistant_reply, self.config

            if self.config.speak_mode:
                say_text(f"I want to execute {cmd}")

        except Exception as e:
            logger.error("Error: \n", str(e))

        return cmd

    def interact_with_assistant(
        self,
    ) -> Tuple[InteractionResult, ChatModelResponse, Dict]:
        status = InteractionResult.OK

        def upd() -> None:
            logger.info("Soft interrupt")
            nonlocal status
            status = InteractionResult.SoftInterrupt

        # Send message to AI, get response
        assistant_reply_json = {}  # type: ignore
        try:
            with Spinner(
                "Thinking... (q to stop immediately, <space> to stop afterwards) ",
                interruptable=True,
                on_soft_interrupt=upd,
                plain_output=self.config.plain_output,
            ) as spn:
                # convert this call to thread

                assistant_reply = asyncio.run(
                    self.async_task_and_spin(
                        spn,
                        chat_with_ai,
                        (
                            self.config,
                            self,
                            self.system_prompt,
                            self.triggering_prompt,
                            self.fast_token_limit,
                            self.config.fast_llm_model,
                        ),
                    )
                )
        except SpinnerInterrupted:
            logger.warn("Task canceled")
            assistant_reply_json = {}
            assistant_reply = None
            status = InteractionResult.HardInterrupt

        if assistant_reply is not None:
            try:
                assistant_reply_json = extract_json_from_response(
                    assistant_reply.content
                )
                validate_json(assistant_reply_json, self.config)
            except json.JSONDecodeError as e:
                logger.error(f"Exception while validating assistant reply JSON: {e}")
                assistant_reply_json = {}
                status = InteractionResult.ExceptionInValidation

        if assistant_reply_json != {}:
            for plugin in self.config.plugins:
                if not plugin.can_handle_post_planning():
                    continue
                assistant_reply_json = plugin.post_planning(assistant_reply_json)

        return status, assistant_reply, assistant_reply_json
