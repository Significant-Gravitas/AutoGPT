import json
import signal
import sys
from datetime import datetime

from colorama import Fore, Style

from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.llm.base import MessageCycle
from autogpt.llm.chat import chat_with_ai
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS
from autogpt.llm.utils import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.memory.message_history import MessageHistory
from autogpt.memory.vector import VectorMemory
from autogpt.models.command_function import CommandFunction
from autogpt.models.command_registry import CommandRegistry
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace

CONTINUE_STRING = "What do you want to do next?"


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
        workspace_directory: str,
        config: Config,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.history = MessageHistory(self)
        self.next_action_count = next_action_count
        self.command_registry = command_registry
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

    def start_interaction_loop(self):
        # Avoid circular imports
        from autogpt.app import execute_command

        # Interaction Loop
        self.cycle_count = 0
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
                [m.raw() for m in self.history.messages],
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

            # STEP 1: Get AI Response
            with Spinner("Thinking... ", plain_output=self.config.plain_output):
                assistant_reply = chat_with_ai(
                    config=self.config,
                    agent=self,
                    system_prompt=self.system_prompt,
                    triggering_prompt=self.triggering_prompt,
                    token_limit=self.fast_token_limit,
                    model=self.config.fast_llm_model,
                    functions=self.get_functions_from_commands(),
                )

            reply_content = assistant_reply.content
            reply_content_json = {}
            if reply_content:
                # Sometimes the content can be duplicated, and this will contain 2 objects in one string
                # TODO: Why is this?
                reply_content = reply_content.split("}\n{")[0]

                # TODO: Sometimes the content doesn't close the last bracket. Why?
                if not reply_content.endswith("}"):
                    reply_content += "}"

                for plugin in self.config.plugins:
                    if not plugin.can_handle_post_planning():
                        continue
                    reply_content = plugin.post_planning(reply_content)

                try:
                    reply_content_json = json.loads(reply_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Could not decode response JSON")
                    logger.debug(f"Invalid response JSON: {reply_content_json}")
                    continue

                print_assistant_thoughts(
                    self.ai_name, reply_content_json, self.config.speak_mode
                )
                self.log_cycle_handler.log_cycle(
                    self.ai_config.ai_name,
                    self.created_at,
                    self.cycle_count,
                    reply_content_json,
                    NEXT_ACTION_FILE_NAME,
                )
                # TODO: Validate
            else:
                logger.warn("AI Response did not include content")

            function_call = assistant_reply.function_call
            if function_call:
                # TODO: What should happen when there's no function call? The AI does this sometimes. Maybe when it
                #  thinks it's done
                command_name = function_call.get("name")
                # FIXME: Some call is sending a string, some arent?
                arguments = function_call.get("arguments")
                if type(arguments) == str:
                    try:
                        arguments = json.loads(function_call.get("arguments"))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Error: Could not parse arguments (probably due to improper escaping)"
                        )
                        logger.debug(str(function_call.get(arguments)))
                        arguments = {}

            if self.config.speak_mode:
                say_text(f"I want to execute {command_name}")

            # First log new-line so user can differentiate sections better in console
            logger.typewriter_log("\n")
            logger.typewriter_log(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
            )

            # Step 2: Gather user input
            if not self.config.continuous_mode and self.next_action_count == 0:
                # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.info(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands, "
                    "'n' to exit program, or enter feedback for "
                    f"{self.ai_name}..."
                )
                # User input loop, continue until the authorise key ('y') is pressed
                while True:
                    if self.config.chat_messages_enabled:
                        console_input = clean_input("Waiting for your response...")
                    else:
                        console_input = clean_input(
                            Fore.MAGENTA + "Input:" + Style.RESET_ALL
                        )

                    if console_input.lower().strip() == self.config.authorise_key:
                        user_input = CONTINUE_STRING
                        break

                    # We didn't get the authorise key
                    if console_input.lower().strip() == "":
                        logger.warn("Invalid input format.")
                        continue

                    if console_input.lower().startswith(
                        f"{self.config.authorise_key} -"
                    ):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            user_input = CONTINUE_STRING
                        except ValueError:
                            logger.warn(
                                "Invalid input format. Please enter 'y -n' where n is"
                                " the number of continuous tasks."
                            )
                            continue
                        break

                    if console_input.lower() == self.config.exit_key:
                        user_input = "EXIT"
                        break

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

                if user_input == CONTINUE_STRING:
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

            # Step 2: Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                command_result = (
                    text_result
                ) = f"Could not execute command: {command_name}"
            elif command_name == "human_feedback":
                command_result = text_result = f"Human feedback: {user_input}"
            else:
                for plugin in self.config.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(
                        command_name, arguments
                    )

                command_result = execute_command(
                    self.command_registry,
                    command_name,
                    arguments,
                    agent=self,
                )
                # TODO: Something should change here
                text_result = f"Command {command_name} returned: " f"{command_result}"

            result_tlength = count_string_tokens(
                str(command_result), self.config.fast_llm_model
            )
            memory_tlength = count_string_tokens(
                str(self.history.summary_message()), self.config.fast_llm_model
            )
            if result_tlength + memory_tlength + 600 > self.fast_token_limit:
                text_result = f"Failure: command {command_name} returned too much output. \
                    Do not execute this command again with the same arguments."

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                text_result = plugin.post_command(command_name, text_result)
            if self.next_action_count > 0:
                self.next_action_count -= 1

            # Check if there's a result from the command append it to the message
            # history
            if not text_result:
                text_result = "Unable to execute command"
                logger.typewriter_log(
                    "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                )

            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, text_result)

            # Step 4: Push results to history
            message_cycle = MessageCycle.construct(
                triggering_prompt=self.triggering_prompt,
                ai_response=reply_content,
                user_input=user_input,
                command_result=command_result,
                command_name=command_name,
                command_arguments=arguments,
            )
            self.history.add(message_cycle)

    def _resolve_pathlike_command_args(self, command_args):
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in command_args:
                    command_args[pathlike] = str(
                        self.workspace.get_path(command_args[pathlike])
                    )
        return command_args

    def get_functions_from_commands(self) -> list[CommandFunction]:
        """Get functions from the commands. "functions" in this context refers to OpenAI functions

        see https://platform.openai.com/docs/guides/gpt/function-calling
        """
        functions = []
        for command in self.command_registry.commands.values():
            properties = {}
            required = []

            for argument in command.arguments:
                properties[argument.name] = {
                    "type": argument.type,
                    "description": argument.description,
                }
                if argument.required:
                    required.append(argument.name)

            parameters = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
            functions.append(
                CommandFunction(
                    name=command.name,
                    description=command.description,
                    parameters=parameters,
                )
            )

        return functions
