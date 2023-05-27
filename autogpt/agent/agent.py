import signal
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from colorama import Fore, Style

from autogpt.app import execute_command, get_command_message
from autogpt.commands.command import CommandRegistry
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm.base import ChatSequence, CommandError, CommandMessage
from autogpt.llm.chat import chat_with_ai, create_chat_completion
from autogpt.llm.utils import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
    SUPERVISOR_FEEDBACK_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.memory.message_history import MessageHistory
from autogpt.memory.vector import VectorMemory
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


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
            Determine which next command to use, and respond using the format specified
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
        config: AIConfig,
        system_prompt: str,
        triggering_prompt: str,
        workspace_directory: str,
    ):
        self.cfg = Config()
        self.ai_name = ai_name
        self.memory = memory
        self.history = MessageHistory(self)
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.config = config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, self.cfg.restrict_to_workspace)
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cycle_count = 0
        self.log_cycle_handler = LogCycleHandler()

        self.error_count = 0
        self.errors: Dict[str, list[CommandError]] = defaultdict(list)

    def start_interaction_loop(self) -> None:
        # Interaction Loop
        self.cycle_count = 0
        self.error_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)

        while True:
            # Discontinue if continuous limit is reached
            self.cycle_count += 1
            self.log_cycle_handler.log_count_within_cycle = 0
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                [m.raw() for m in self.history],
                FULL_MESSAGE_HISTORY_FILE_NAME,
            )

            if self._continuous_limit_reached():
                logger.typewriter_log(
                    "Continuous Limit Reached: ",
                    Fore.YELLOW,
                    f"{self.cfg.continuous_limit}",
                )
                break

            # Send message to AI, get response
            with Spinner("Thinking... ", plain_output=self.cfg.plain_output):
                assistant_reply = chat_with_ai(
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    self.cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = self._convert_assistant_reply_to_json(
                assistant_reply
            )

            command_msg = self._parse_command_and_arguments(assistant_reply_json)

            if isinstance(command_msg, CommandMessage):
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_msg.name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{command_msg.args}{Style.RESET_ALL}",
                )
            else:
                self._add_error(command_msg)
                logger.typewriter_log(
                    "COULD NOT PARSE COMMAND: ",
                    Fore.RED,
                    f"{command_msg.msg}",
                )

            if not self.cfg.continuous_mode and self.next_action_count == 0:
                command_msg = self._get_user_input(command_msg, assistant_reply_json)

            if self.cfg.continuous_mode and self._error_threshold_reached():
                command_msg = self._handle_self_feedback(assistant_reply_json)

            if self.next_action_count > 0:
                logger.typewriter_log(
                    f"{Fore.CYAN}AUTHORISED COMMANDS LEFT: {Style.RESET_ALL}{self.next_action_count}"
                )

            if isinstance(command_msg, CommandMessage):
                if command_msg.user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )

                if command_msg.user_input == "EXIT":
                    logger.info("Exiting...")
                    break

                result = self._handle_command_message(command_msg)
            else:
                result = None

            self._append_result_to_full_message_history(result)

    def _convert_assistant_reply_to_json(self, assistant_reply: str) -> Dict:
        assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_post_planning():
                continue

            assistant_reply_json = plugin.post_planning(assistant_reply_json)

        validate_json(assistant_reply_json, LLM_DEFAULT_RESPONSE_FORMAT)

        return assistant_reply_json

    def _parse_command_and_arguments(
        self, assistant_reply_json: Dict
    ) -> CommandMessage | CommandError:
        # Get command name and arguments
        try:
            print_assistant_thoughts(
                self.ai_name, assistant_reply_json, self.cfg.speak_mode
            )
            command_msg = get_command_message(assistant_reply_json)

            if self.cfg.speak_mode:
                say_text(f"I want to execute {command_msg.name}")

            command_msg.args = self._resolve_pathlike_command_args(command_msg.args)

            return command_msg
        except Exception as e:
            logger.error("Error: \n", str(e))
            return CommandError("invalid_command", {}, str(e))
        finally:
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                assistant_reply_json,
                NEXT_ACTION_FILE_NAME,
            )

    def _get_console_input(self) -> str:
        if self.cfg.chat_messages_enabled:
            return clean_input("Waiting for your response...")

        return clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)

    def _get_user_input(
        self, command_msg: CommandMessage | CommandError, assistant_reply_json: Dict
    ) -> CommandMessage | CommandError:
        # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
        # Get key press: Prompt the user to press enter to continue or escape
        # to exit

        if isinstance(command_msg, CommandMessage):
            logger.info(
                "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands, "
                "'n' to exit program, or enter feedback for "
                f"{self.ai_name}..."
            )

            while True:
                console_input = self._get_console_input()

                if console_input.lower().strip() == self.cfg.authorise_key:
                    command_msg.user_input = "GENERATE NEXT COMMAND JSON"
                    return command_msg
                elif console_input.lower().strip() == "s":
                    return self._handle_self_feedback(assistant_reply_json)
                elif console_input.lower().strip() == "":
                    logger.warn("Invalid input format.")
                    continue
                elif console_input.lower().startswith(f"{self.cfg.authorise_key} -"):
                    try:
                        self.next_action_count = abs(int(console_input.split(" ")[1]))
                        command_msg.user_input = "GENERATE NEXT COMMAND JSON"
                    except ValueError:
                        logger.warn(
                            "Invalid input format. Please enter 'y -n' where n is"
                            " the number of continuous tasks."
                        )
                        continue

                    return command_msg
                elif console_input.lower() == self.cfg.exit_key:
                    command_msg.name = "exit"
                    command_msg.user_input = "EXIT"
                    return command_msg
                else:
                    command_msg.user_input = console_input
                    command_msg.name = "human_feedback"
                    self.log_cycle_handler.log_cycle(
                        self.config.ai_name,
                        self.created_at,
                        self.cycle_count,
                        console_input,
                        USER_INPUT_FILE_NAME,
                    )
                    return command_msg
        else:
            logger.info(
                "Command is invalid. Enter 's' to run self-feedback commands, "
                f"'n' to exit program, or enter feedback for {self.ai_name}..."
            )

            while True:
                console_input = self._get_console_input()

                if console_input.lower().strip() == "s":
                    return self._handle_self_feedback(assistant_reply_json)
                elif console_input.lower().strip() == "":
                    logger.warn("Invalid input format.")
                    continue
                elif console_input.lower() == self.cfg.exit_key:
                    return CommandMessage("exit", {}, "EXIT")
                else:
                    command_msg = CommandMessage("human_feedback", {}, console_input)
                    command_msg.user_input = console_input
                    command_msg.name = "human_feedback"
                    self.log_cycle_handler.log_cycle(
                        self.config.ai_name,
                        self.created_at,
                        self.cycle_count,
                        console_input,
                        USER_INPUT_FILE_NAME,
                    )
                    return command_msg

    def _handle_command_message(self, command_msg: CommandMessage) -> str:
        command_name = command_msg.name
        arguments = command_msg.args
        user_input = command_msg.user_input

        if command_name == "human_feedback":
            return f"Human feedback: {user_input}"

        if command_name == "self_feedback":
            return f"Self feedback: {user_input}"

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_pre_command():
                continue

            command_name, arguments = plugin.pre_command(command_name, arguments)

        command_result = execute_command(
            self.command_registry,
            command_name,
            arguments,
            self.config.prompt_generator,
            config=self.cfg,
        )

        return self._validate_command_result(command_msg, str(command_result))

    def _validate_command_result(
        self, command_msg: CommandMessage, command_result: str
    ) -> str:
        command_name = command_msg.name
        arguments = command_msg.args

        if self._is_output_too_large(command_result):
            self._add_error(CommandError(command_name, arguments, "Too much output."))
            result = (
                f"Failure: command '{command_name}' returned too much "
                f"output. Do not execute this command again with the same arguments."
            )
        elif is_command_result_an_error(command_result):
            self._add_error(CommandError(command_name, arguments, command_result))
            result = (
                f"Failure: command '{command_name}' returned the following "
                f"error: '{command_result}'. Do not execute this command "
                f"again with the same arguments."
            )
        else:
            result = f"Command '{command_name}' returned: {command_result}"

        for plugin in self.cfg.plugins:
            if not plugin.can_handle_post_command():
                continue

            result = plugin.post_command(command_name, result)

        if self.next_action_count > 0:
            self.next_action_count -= 1

        return result

    def _append_result_to_full_message_history(self, result: Optional[str]) -> None:
        if result is not None:
            self.history.add("system", result, "action_result")
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
        else:
            self.history.add("system", "Unable to execute command", "action_result")
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

    def _error_threshold_reached(self) -> bool:
        return self.error_count >= self.cfg.error_threshold > 0

    def _is_output_too_large(self, command_result: str) -> bool:
        result_tlength = count_string_tokens(
            str(command_result), self.cfg.fast_llm_model
        )
        memory_tlength = count_string_tokens(
            str(self.history.summary_message()), self.cfg.fast_llm_model
        )

        return result_tlength + memory_tlength + 600 > self.cfg.fast_token_limit

    def _add_error(self, command_err: CommandError) -> None:
        self.errors[command_err.name].append(command_err)
        self.error_count += 1

    def _handle_self_feedback(
        self,
        assistant_reply_json: Dict[Any, Any],
    ) -> CommandMessage:
        self.error_count = 0

        logger.typewriter_log(
            "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
            Fore.GREEN,
            "",
        )
        thoughts = assistant_reply_json.get("thoughts", {})

        with Spinner("Getting self feedback... "):
            self_feedback_resp = self._get_self_feedback_from_ai(
                thoughts, self.cfg.fast_llm_model
            )

        logger.typewriter_log(
            f"SELF FEEDBACK: {self_feedback_resp}",
            Fore.YELLOW,
            "",
        )
        user_input = self_feedback_resp
        command_name = "self_feedback"

        return CommandMessage(command_name, {}, user_input)

    def _resolve_pathlike_command_args(
        self, command_args: Dict[str, Any]
    ) -> Dict[str, str]:
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in command_args:
                    command_args[pathlike] = str(
                        self.workspace.get_path(command_args[pathlike])
                    )
        return command_args

    def _get_self_feedback_from_ai(
        self,
        thoughts: dict,
        llm_model: str,
    ) -> str:
        feedback_prompt = self.construct_self_feedback_prompt(thoughts)

        prompt = ChatSequence.for_model(llm_model)
        prompt.add("user", feedback_prompt)

        self.log_cycle_handler.log_cycle(
            self.config.ai_name,
            self.created_at,
            self.cycle_count,
            prompt.raw(),
            PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
        )

        feedback = create_chat_completion(prompt)

        self.log_cycle_handler.log_cycle(
            self.config.ai_name,
            self.created_at,
            self.cycle_count,
            feedback,
            SUPERVISOR_FEEDBACK_FILE_NAME,
        )

        return feedback

    def construct_self_feedback_prompt(
        self,
        thoughts: Dict[str, str],
    ) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
              including the ai_name, ai_role, ai_goals, and api_budget.
        """

        prompt_start = (
            f"Below is a message from me, an AI Agent, assuming the role of "
            f"{self.config.ai_role} Whilst keeping knowledge of my slight "
            f"limitations as an AI Agent, please evaluate my overall goals, "
            f"constraints, commands, thought process, reasoning, and plan."
        )

        # Construct self feedback prompt
        full_prompt = f"{prompt_start}\n\nGoals:\n"

        for i, goal in enumerate(self.config.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"\n{self.generate_feedback_prompt_string(thoughts)}"

        return full_prompt

    def generate_feedback_prompt_string(self, thoughts: Dict[str, str]) -> str:
        """
        Generate a prompt string based on the constraints, commands, resources,
            and performance evaluations.

        Returns:
            str: The generated prompt string.
        """
        prompt_generator = self.config.prompt_generator
        thought = thoughts.get("text", "")
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")

        full_prompt = (
            f"Constraints:\n"
            f"{prompt_generator.generate_numbered_list(prompt_generator.constraints)}\n\n"
            "Commands:\n"
            f"{prompt_generator.generate_numbered_list(prompt_generator.commands, item_type='command')}\n\n"
            f"Thoughts: {thought}\n"
            f"Reasoning: {reasoning}\n"
            f"Plan:\n{plan}\n\n"
            "You should respond with a concise paragraph that contains any "
            "improvements to my overall thoughts, reasoning, and plan. Based "
            "on these improvements, provide the most likely course of action "
            "that will get us closer to our goals."
        )

        if self.errors is not None:
            full_prompt += (
                f"\nThe course of action provided must also not repeat the "
                f"errors below by using these combinations of commands and "
                f"arguments:\n"
                f"{self._generate_error_list()}"
            )

        # prompt = ChatSequence.for_model(llm_model)
        # prompt.add("user", feedback_prompt + feedback_thoughts)
        #
        # self.log_cycle_handler.log_cycle(
        #     self.config.ai_name,
        #     self.created_at,
        #     self.cycle_count,
        #     prompt.raw(),
        #     PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
        # )
        #
        # feedback = create_chat_completion(prompt)

        return full_prompt

    def _generate_error_list(self) -> str:
        errors = []

        for command, error_list in self.errors.items():
            if command == "invalid_command":
                continue

            for error in error_list:
                errors.append(error)

        error_strings = []

        for command_error in errors:
            args_string = ", ".join(
                f'"{key}": "{value}"' for key, value in command_error.args.items()
            )
            error_strings.append(
                f"{command_error.name}, arguments: {args_string}, "
                f"error message: {command_error.msg}"
            )

        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(error_strings))

    def _continuous_limit_reached(self) -> bool:
        return (
            self.cfg.continuous_mode
            and 0 < self.cfg.continuous_limit < self.cycle_count
        )

    def _signal_handler(self, signum, frame) -> None:
        if self.next_action_count == 0:
            sys.exit()
        else:
            print(
                Fore.RED
                + "Interrupt signal received. Stopping continuous command execution."
                + Style.RESET_ALL
            )
            self.next_action_count = 0


STARTS_WITH_ERROR_STRING = [
    "error",
    "unknown command",
    "traceback",
]

CONTAINS_ERROR_STRING = [
    "no such file or directory",
    "can't open file",
]


def is_command_result_an_error(result: str) -> bool:
    result_lower = result.lower()

    for error_string in STARTS_WITH_ERROR_STRING:
        if result_lower.startswith(error_string):
            return True

    for error_string in CONTAINS_ERROR_STRING:
        if error_string in result_lower:
            return True

    return False
