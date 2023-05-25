import signal
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from colorama import Fore, Style

from autogpt.app import execute_command, get_command
from autogpt.config import Config
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm import (
    Message,
    chat_with_ai,
    create_chat_completion,
    create_chat_message,
)
from autogpt.llm.command_error import CommandError
from autogpt.llm.token_counter import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
    SUPERVISOR_FEEDBACK_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
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
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        command_registry,
        config,
        system_prompt,
        triggering_prompt,
        workspace_directory,
    ) -> None:
        self.cfg = Config()
        self.ai_name = ai_name
        self.memory = memory
        self.summary_memory = (
            "I was created."  # Initial memory necessary to avoid hallucination
        )
        self.last_memory_index = 0
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.ai_config = config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, self.cfg.restrict_to_workspace)
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_cycle_handler = LogCycleHandler()
        self.cycle_count = 0
        self.error_count = 0
        self.errors = []

    def start_interaction_loop(self) -> None:
        # Interaction Loop
        self.cycle_count = 0
        self.error_count = 0
        command_name = None
        arguments = None
        user_input = ""

        # Signal handler for interrupting y -N
        def signal_handler(signum, frame):
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

        while True:
            # Discontinue if continuous limit is reached
            self.cycle_count += 1
            self.log_cycle_handler.log_count_within_cycle = 0
            self.log_cycle_handler.log_cycle(
                self.ai_config.ai_name,
                self.created_at,
                self.cycle_count,
                self.full_message_history,
                FULL_MESSAGE_HISTORY_FILE_NAME,
            )
            if (
                self.cfg.continuous_mode
                and 0 < self.cfg.continuous_limit < self.cycle_count
            ):
                logger.typewriter_log(
                    "Continuous Limit Reached: ",
                    Fore.YELLOW,
                    f"{self.cfg.continuous_limit}",
                )
                break

            # print(f"System prompt: {self.system_prompt}")
            # print(f"Triggering prompt: {self.triggering_prompt}")
            # print(f"Full message history: {self.full_message_history}")

            # Send message to AI, get response
            with Spinner("Thinking... "):
                assistant_reply = chat_with_ai(
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    self.full_message_history,
                    self.memory,
                    self.cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)

            for plugin in self.cfg.plugins:
                if not plugin.can_handle_post_planning():
                    continue

                assistant_reply_json = plugin.post_planning(assistant_reply_json)

            # Print Assistant thoughts
            if assistant_reply_json != {}:
                validate_json(assistant_reply_json, LLM_DEFAULT_RESPONSE_FORMAT)
                # Get command name and arguments
                try:
                    print_assistant_thoughts(
                        self.ai_name, assistant_reply_json, self.cfg.speak_mode
                    )
                    command_name, arguments = get_command(assistant_reply_json)

                    if self.cfg.speak_mode:
                        say_text(f"I want to execute {command_name}")

                    arguments = self._resolve_pathlike_command_args(arguments)

                except Exception as e:
                    logger.error("Error: \n", str(e))

            self.log_cycle_handler.log_cycle(
                self.ai_config.ai_name,
                self.created_at,
                self.cycle_count,
                assistant_reply_json,
                NEXT_ACTION_FILE_NAME,
            )

            logger.typewriter_log(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
            )

            # TODO: Validate Command Name & Arguments
            # TODO: If not valid, send error to GPT and try again

            if not self.cfg.continuous_mode and self.next_action_count == 0:
                # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                logger.info(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands, "
                    "'n' to exit program, or enter feedback for "
                    f"{self.ai_name}..."
                )
                while True:
                    if self.cfg.chat_messages_enabled:
                        console_input = clean_input("Waiting for your response...")
                    else:
                        console_input = clean_input(
                            Fore.MAGENTA + "Input:" + Style.RESET_ALL
                        )
                    if console_input.lower().strip() == self.cfg.authorise_key:
                        user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().strip() == "s":
                        user_input, command_name = self._handle_self_feedback(
                            assistant_reply_json
                        )
                        break
                    elif console_input.lower().strip() == "":
                        logger.warn("Invalid input format.")
                        continue
                    elif console_input.lower().startswith(
                        f"{self.cfg.authorise_key} -"
                    ):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            logger.warn(
                                "Invalid input format. Please enter 'y -n' where n is"
                                " the number of continuous tasks."
                            )
                            continue
                        break
                    elif console_input.lower() == self.cfg.exit_key:
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
                    break
            elif self.cfg.continuous_mode and self._error_threshold_reached():
                user_input, command_name = self._handle_self_feedback(
                    assistant_reply_json
                )
            else:
                # Print authorized commands left value
                logger.typewriter_log(
                    f"{Fore.CYAN}AUTHORISED COMMANDS LEFT: {Style.RESET_ALL}{self.next_action_count}"
                )

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                self._add_error(None, {}, command_name + " " + str(arguments))
                result = (
                    f"Command {command_name} threw the following error: {arguments}"
                )
            elif command_name == "human_feedback":
                result = f"Human feedback: {user_input}"
            elif command_name == "self_feedback":
                result = f"Self feedback: {user_input}"
            else:
                for plugin in self.cfg.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(
                        command_name, arguments
                    )
                command_result = execute_command(
                    self.command_registry,
                    command_name,
                    arguments,
                    self.ai_config.prompt_generator,
                )

                if self._is_output_too_large(command_result):
                    self._add_error(command_name, arguments, "Too much output.")
                    result = f"Failure: command {command_name} returned too much output. \
                        Do not execute this command again with the same arguments."
                elif is_command_result_an_error(str(command_result)):
                    self._add_error(command_name, arguments, command_result)
                    result = (
                        f"Failure: command {command_name} returned the "
                        f"following error: '{command_result}'. Avoid this by not "
                        f"executing this command again with the same arguments."
                    )
                else:
                    result = f"Command {command_name} returned: {command_result}"

                for plugin in self.cfg.plugins:
                    if not plugin.can_handle_post_command():
                        continue
                    result = plugin.post_command(command_name, result)
                if self.next_action_count > 0:
                    self.next_action_count -= 1

            # Check if there's a result from the command append it to the message
            # history
            if result is not None:
                self.full_message_history.append(create_chat_message("system", result))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                self.full_message_history.append(
                    create_chat_message("system", "Unable to execute command")
                )
                logger.typewriter_log(
                    "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                )

    def _error_threshold_reached(self) -> bool:
        return self.error_count >= self.cfg.error_threshold > 0

    def _is_output_too_large(self, command_result: Any) -> bool:
        result_tlength = count_string_tokens(
            str(command_result), self.cfg.fast_llm_model
        )
        memory_tlength = count_string_tokens(
            str(self.summary_memory), self.cfg.fast_llm_model
        )

        return result_tlength + memory_tlength + 600 > self.cfg.fast_token_limit

    def _add_error(self, command: Optional[str], arguments: dict, msg: str) -> None:
        self.error_count += 1
        self.errors.append(CommandError(command, arguments, msg))

    def _handle_self_feedback(
        self, assistant_reply_json: Dict[Any, Any]
    ) -> tuple[str, str]:
        self.error_count = 0

        logger.typewriter_log(
            "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
            Fore.GREEN,
            "",
        )
        thoughts = assistant_reply_json.get("thoughts", {})

        prev_error = self.errors[-1] if self.errors else None

        with Spinner("Getting self feedback... "):
            messages, feedback = self._get_self_feedback_from_ai(
                thoughts,
                prev_error,
            )

        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            messages,
            PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
        )

        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            feedback,
            SUPERVISOR_FEEDBACK_FILE_NAME,
        )

        logger.typewriter_log(
            f"SELF FEEDBACK: {feedback}",
            Fore.YELLOW,
            "",
        )
        user_input = feedback
        command_name = "self_feedback"

        return user_input, command_name

    def _resolve_pathlike_command_args(self, command_args) -> Dict[str, str]:
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
        prev_error: Optional[CommandError],
    ) -> Tuple[dict[str, str], str]:
        full_prompt = self.construct_self_feedback_prompt(thoughts, prev_error)

        print(f"Full feedback prompt: {full_prompt}")

        feedback_prompt = {"role": "user", "content": full_prompt}
        messages = [Message(**feedback_prompt)]

        model = self.cfg.fast_llm_model  # TODO: Change model from hardcode to
        # argument
        assistant_reply = create_chat_completion(
            model=model,
            messages=messages,
        )

        return feedback_prompt, assistant_reply

    def construct_self_feedback_prompt(
        self,
        thoughts: dict[str, str],
        prev_error: Optional["CommandError"],
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
            f"{self.ai_config.ai_role} Whilst keeping knowledge of my slight "
            f"limitations "
            f"as an AI Agent, please evaluate my overall goals, constraints, "
            f"commands, thought process, reasoning, and plan."
        )

        if prev_error is not None:
            prompt_start += (
                f" Also know that I am reaching out for feedback because the "
                f"previous command and arguments failed to accomplish my "
                f"latest task, resulting in an error message. This must also "
                f"be taken into consideration in your response so that I do "
                f"not repeat the same mistakes.\n\n"
                f"Previous Command: {prev_error.command}\n"
                f"Previous Arguments: {prev_error.arguments}\n"
                f"Error Message: '{prev_error.message}'"
            )

        # Construct self feedback prompt
        full_prompt = f"{prompt_start}\n\nGOALS:\n"

        for i, goal in enumerate(self.ai_config.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"

        full_prompt += f"{self.generate_feedback_prompt_string(thoughts)}"
        return full_prompt

    def generate_feedback_prompt_string(self, thoughts: dict[str, str]) -> str:
        """
        Generate a prompt string based on the constraints, commands, resources,
            and performance evaluations.

        Returns:
            str: The generated prompt string.
        """
        prompt_generator = self.ai_config.prompt_generator
        thought = thoughts.get("text", "")
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")

        return (
            f"Constraints:\n"
            f"{prompt_generator.generate_numbered_list(prompt_generator.constraints)}\n\n"
            "Commands:\n"
            f"{prompt_generator.generate_numbered_list(prompt_generator.commands, item_type='command')}\n\n"
            f"Thoughts: {thought}\n"
            f"Reasoning: {reasoning}\n"
            f"Plan:\n{plan}\n\n"
            "You should respond with a concise paragraph that contains any "
            "improvements to my overall thoughts, reasoning, and plan. Based "
            "on these improvements, include a list of possible actions "
            "that will get us closer to our goals, while avoiding previous "
            "errors and only using the commands provided with valid "
            "arguments. Order these options from most likely to be effective "
            "to least."
        )


def is_command_result_an_error(result: str) -> bool:
    err_strs = [
        "error",
        "unknown command",
        "traceback",
        "no such file or directory",
        "can't open file",
    ]
    result_lower = result.lower()

    for err_str in err_strs:
        if err_str in result_lower:
            return True

    return False
