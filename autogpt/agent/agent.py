from datetime import datetime

from colorama import Fore, Style

from autogpt.app import execute_command, get_command
from autogpt.commands.command import CommandRegistry
from autogpt.config import AIConfig, Config
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm import chat_with_ai, create_chat_completion, create_chat_message
from autogpt.llm.base import Message as ChatMessage
from autogpt.llm.token_counter import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.memory.base import MemoryProviderSingleton
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input, send_chat_message_to_user
from autogpt.workspace import Workspace

CFG = Config()


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
        autonomous_cycles_budget: The maximum number of autonomous cycles to execute.
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
        memory: MemoryProviderSingleton,
        full_message_history: list[ChatMessage],
        autonomous_cycles_budget: int,
        command_registry: CommandRegistry,
        config: AIConfig,
        system_prompt: str,
        triggering_prompt: str,
        workspace_directory: str,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.summary_memory = (
            "I was created."  # Initial memory necessary to avoid hallucination
        )
        self.last_memory_index = 0
        self.full_message_history = full_message_history
        self.autonomous_cycles_remaining = autonomous_cycles_budget
        self.command_registry = command_registry
        self.config = config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, CFG.restrict_to_workspace)

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cycle_count = 0
        self.log_cycle_handler = LogCycleHandler()

    def start_interaction_loop(self):
        # Interaction Loop
        cfg = Config()
        self.cycle_count = 0
        command_name = None
        arguments = None
        user_input = ""

        while True:
            # Discontinue if continuous limit is reached
            self.cycle_count += 1
            self.log_cycle_handler.log_count_within_cycle = 0
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                self.full_message_history,
                FULL_MESSAGE_HISTORY_FILE_NAME,
            )
            if (
                CFG.continuous_mode
                and CFG.continuous_limit > 0
                and self.cycle_count > CFG.continuous_limit
            ):
                logger.typewriter_log(
                    "Continuous Limit Reached: ", Fore.YELLOW, f"{CFG.continuous_limit}"
                )
                send_chat_message_to_user(
                    f"Continuous Limit Reached: \n {CFG.continuous_limit}"
                )
                break
            # Send message to AI, get response
            with Spinner("Thinking... "):
                assistant_reply = chat_with_ai(
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    self.full_message_history,
                    self.memory,
                    CFG.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)
            for plugin in CFG.plugins:
                if not plugin.can_handle_post_planning():
                    continue
                assistant_reply_json = plugin.post_planning(assistant_reply_json)

            # Print Assistant thoughts
            if assistant_reply_json != {}:
                validate_json(assistant_reply_json, LLM_DEFAULT_RESPONSE_FORMAT)
                # Get command name and arguments
                try:
                    print_assistant_thoughts(
                        self.ai_name, assistant_reply_json, CFG.speak_mode
                    )
                    command_name, arguments = get_command(assistant_reply_json)
                    if CFG.speak_mode:
                        say_text(f"I want to execute {command_name}")

                    arguments = self._resolve_pathlike_command_args(arguments)

                except Exception as e:
                    logger.error("Error: \n", str(e))
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                assistant_reply_json,
                NEXT_ACTION_FILE_NAME,
            )

            if self.should_prompt_user:
                # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )
                print(
                    self.user_feedback_prompt,
                    flush=True,
                )
                while True:
                    if CFG.chat_messages_enabled:
                        console_input = clean_input("Waiting for your response...")
                    else:
                        console_input = clean_input(
                            Fore.MAGENTA + "Input:" + Style.RESET_ALL
                        )
                    if console_input.lower().strip() == cfg.authorise_key:
                        user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().strip() == "s":
                        logger.typewriter_log(
                            "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
                            Fore.GREEN,
                            "",
                        )
                        thoughts = assistant_reply_json.get("thoughts", {})
                        self_feedback_resp = self.get_self_feedback(
                            thoughts, CFG.fast_llm_model
                        )
                        logger.typewriter_log(
                            f"SELF FEEDBACK: {self_feedback_resp}",
                            Fore.YELLOW,
                            "",
                        )
                        if self_feedback_resp[0].lower().strip() == cfg.authorise_key:
                            user_input = "GENERATE NEXT COMMAND JSON"
                        else:
                            user_input = self_feedback_resp
                        break
                    elif console_input.lower().strip() == "":
                        print("Invalid input format.")
                        continue
                    elif console_input.lower().startswith(f"{cfg.authorise_key} -"):
                        try:
                            self.autonomous_cycles_remaining = abs(
                                int(console_input.split(" ")[1])
                            )
                            user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            print(
                                f"Invalid input format. Please enter '{cfg.authorise_key} -N' where N is"
                                " the number of continuous tasks."
                            )
                            continue
                        break
                    elif console_input.lower() == cfg.exit_key:
                        user_input = "EXIT"
                        break
                    else:
                        user_input = console_input
                        command_name = "human_feedback"
                        break
                    console_input = clean_input(
                        Fore.MAGENTA + "Input:" + Style.RESET_ALL
                    )

                    (
                        new_command_name,
                        self.autonomous_cycles_remaining,
                        user_input,
                    ) = self.determine_next_command(console_input)
                    # If there was a parsing error, go back to the prompt
                    # Otherwise if new command name is not None, update command name
                    if new_command_name == "input error":
                        print(user_input)
                        continue
                    elif new_command_name is not None:
                        command_name = new_command_name
                    break

                # Prompt for feedback on the self feedback
                if command_name == "self_feedback":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
                        Fore.GREEN,
                        "",
                        0,
                    )
                    thoughts = assistant_reply_json.get("thoughts", {})
                    self_feedback_resp = self.get_self_feedback(
                        thoughts, cfg.fast_llm_model
                    )
                    logger.typewriter_log(
                        f"SELF FEEDBACK: {self_feedback_resp}",
                        Fore.YELLOW,
                        "",
                    )
                    if self_feedback_resp[0].lower().strip() == "y":
                        user_input = "GENERATE NEXT COMMAND JSON"
                    else:
                        user_input = self_feedback_resp

                elif user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )
                elif user_input == "EXIT":
                    send_chat_message_to_user("Exiting...")
                    print("Exiting...", flush=True)
                    break
            else:
                # Print command
                send_chat_message_to_user(
                    "NEXT ACTION: \n " + f"COMMAND = {command_name} \n "
                    f"ARGUMENTS = {arguments}"
                )

                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}"
                    f"  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                result = (
                    f"Command {command_name} threw the following error: {arguments}"
                )
            elif command_name == "human_feedback":
                result = f"Human feedback: {user_input}"
                self.log_cycle_handler.log_cycle(
                    self.config.ai_name,
                    self.created_at,
                    self.cycle_count,
                    user_input,
                    USER_INPUT_FILE_NAME,
                )
            elif command_name == "self_feedback":
                result = f"Self feedback: {user_input}"
            else:
                for plugin in CFG.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(
                        command_name, arguments
                    )
                command_result = execute_command(
                    self.command_registry,
                    command_name,
                    arguments,
                    self.config.prompt_generator,
                )
                result = f"Command {command_name} returned: " f"{command_result}"

                # mitigate context overflow errors
                result_tlength = count_string_tokens(
                    str(command_result), cfg.fast_llm_model
                )
                memory_tlength = count_string_tokens(
                    str(self.summary_memory), cfg.fast_llm_model
                )
                if result_tlength + memory_tlength + 600 > cfg.fast_token_limit:
                    result = f"Failure: command {command_name} returned too much output. \
                        Do not execute this command again with the same arguments."

                for plugin in CFG.plugins:
                    if not plugin.can_handle_post_command():
                        continue
                    result = plugin.post_command(command_name, result)
                if self.autonomous_cycles_remaining > 0:
                    self.autonomous_cycles_remaining -= 1

                # Check if there's a result from the command append it to the message
                # history
                if result is not None:
                    self.full_message_history.append(
                        create_chat_message("system", result)
                    )
                    logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
                else:
                    self.full_message_history.append(
                        create_chat_message("system", "Unable to execute command")
                    )
                    logger.typewriter_log(
                        "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                    )

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

    def get_self_feedback(self, thoughts: dict, llm_model: str) -> str:
        """Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.
        Args:
            thoughts (dict): A dictionary containing thought elements like reasoning,
            plan, thoughts, and criticism.
        Returns:
            str: A feedback response generated using the provided thoughts dictionary.
        """
        ai_role = self.config.ai_role

        feedback_prompt = (
            f"Below is a message from me, an AI Agent, assuming the role of {ai_role}. "
            "Whilst keeping knowledge of my slight limitations as an AI Agent, evaluate "
            "my thought process, reasoning, and plan, and provide a concise paragraph "
            "outlining potential improvements. Consider adding or removing ideas that "
            "do not align with my role and explaining why, prioritizing thoughts based "
            "on their significance, or simply refining my overall thought process."
        )
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")
        thought = thoughts.get("thoughts", "")
        feedback_thoughts = thought + reasoning + plan
        return create_chat_completion(
            [{"role": "user", "content": feedback_prompt + feedback_thoughts}],
            llm_model,
        )

    @property
    def should_prompt_user(self) -> bool:
        return not CFG.continuous_mode and self.autonomous_cycles_remaining == 0

    @property
    def user_feedback_prompt(self) -> str:
        return (
            "Enter 'y' to authorise command, 'y -N' to run N continuous commands, "
            "'s' to run self-feedback commands, "
            "'n' to exit program, or enter feedback for "
            f"{self.ai_name}... "
        )

    @staticmethod
    def determine_next_command(user_input: str):
        command_name = str | None
        autonomous_cycles_remaining = 0

        if user_input.lower().rstrip() == "y":
            command_name = None
            user_input = "GENERATE NEXT COMMAND JSON"
        elif user_input.lower().startswith(f"{CFG.authorise_key} -"):
            try:
                autonomous_cycles_remaining = abs(int(user_input.split(" ")[1]))
                user_input = "GENERATE NEXT COMMAND JSON"
                command_name = None
            except ValueError:
                command_name = "input error"
                user_input = (
                    "Invalid input format. Please enter 'y -n' where n is"
                    " the number of continuous tasks."
                )
        elif user_input.lower().strip() == "":
            command_name = "input error"
            user_input = "Invalid input format."
        elif user_input.lower().strip() == "n":
            command_name = None
            user_input = "EXIT"
        elif user_input.lower().strip() == "s":
            user_input = None
            command_name = "self_feedback"
        else:
            user_input = user_input
            command_name = "human_feedback"

        return command_name, autonomous_cycles_remaining, user_input
