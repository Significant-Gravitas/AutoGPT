from typing import Any, Dict, NoReturn, Tuple, Union

from colorama import Fore, Style

from autogpt.app import execute_command, get_command
from autogpt.chat import chat_with_ai, create_chat_message
from autogpt.config import Config
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import validate_json
from autogpt.llm_utils import create_chat_completion
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input, send_chat_message_to_user
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
    ):
        cfg = Config()
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.config = config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, cfg.restrict_to_workspace)

    def start_interaction_loop(self) -> NoReturn:
        cfg = Config()
        loop_count = 0
        command_name = None
        arguments = None
        user_input = ""

        while True:
            loop_count, cfg, command_name, arguments, user_input = self.interact(
                loop_count, cfg, command_name, arguments, user_input
            )

    def check_continuous(self, cfg, loop_count) -> None:
        if (
            cfg.continuous_mode
            and cfg.continuous_limit > 0
            and loop_count > cfg.continuous_limit
        ):
            logger.typewriter_log(
                "Continuous Limit Reached: ", Fore.YELLOW, f"{cfg.continuous_limit}"
            )
            send_chat_message_to_user(
                f"Continuous Limit Reached: \n {cfg.continuous_limit}"
            )
            exit(0)

    def get_assistant_reply(self, cfg, spinner=False) -> str:
        if spinner:
            with Spinner("Thinking... "):
                return chat_with_ai(
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    self.full_message_history,
                    self.memory,
                    cfg.fast_token_limit,
                )
        return chat_with_ai(
            self,
            self.system_prompt,
            self.triggering_prompt,
            self.full_message_history,
            self.memory,
            cfg.fast_token_limit,
        )

    def handle_post_planning(self, cfg, assistant_reply_json) -> Dict[str, Any]:
        for plugin in cfg.plugins:
            if not plugin.can_handle_post_planning():
                continue
            assistant_reply_json = plugin.post_planning(self, assistant_reply_json)
        return assistant_reply_json

    def resolve_assistant_command(self, cfg, assistant_reply_json):
        command_name = "None"
        arguments = []

        if assistant_reply_json == {}:
            return command_name, arguments

        validate_json(assistant_reply_json, "llm_response_format_1")
        try:
            print_assistant_thoughts(self.ai_name, assistant_reply_json, cfg.speak_mode)
            command_name, arguments = get_command(assistant_reply_json)
            if cfg.speak_mode:
                say_text(f"I want to execute {command_name}")

            send_chat_message_to_user("Thinking... \n")
            arguments = self._resolve_pathlike_command_args(arguments)
        except Exception as e:
            logger.error("Error: \n", str(e))
        return command_name, arguments

    def process_assistant_reply(self, cfg):
        assistant_reply = self.get_assistant_reply(cfg)
        assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)
        assistant_reply_json = self.handle_post_planning(cfg, assistant_reply_json)
        command_name, arguments = self.resolve_assistant_command(
            cfg, assistant_reply_json
        )
        return command_name, arguments, assistant_reply, assistant_reply_json

    def get_console_input(self, cfg) -> str:
        if cfg.chat_messages_enabled:
            return clean_input("Waiting for your response...")
        return clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)

    def process_self_feedback(self, assistant_reply_json, cfg):
        logger.typewriter_log(
            "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
            Fore.GREEN,
            "",
        )
        thoughts = assistant_reply_json.get("thoughts", {})
        self_feedback_resp = self.get_self_feedback(thoughts, cfg.fast_llm_model)
        logger.typewriter_log(
            f"SELF FEEDBACK: {self_feedback_resp}",
            Fore.YELLOW,
            "",
        )
        if self_feedback_resp[0].lower().strip() == "y":
            user_input = "GENERATE NEXT COMMAND JSON"
        else:
            user_input = self_feedback_resp
        return user_input, None, False

    def process_continue_for(self, user_input, console_input):
        try:
            self.next_action_count = abs(int(console_input.split(" ")[1]))
            user_input = "GENERATE NEXT COMMAND JSON"
        except ValueError:
            print(
                "Invalid input format. Please enter 'y -n' where n is"
                " the number of continuous tasks."
            )
            return user_input, None, True
        return user_input, None, False

    def process_console_input(
        self, console_input, cfg, user_input, assistant_reply_json
    ) -> Tuple[str, Union[None, str], bool]:
        if console_input == "y":
            return "GENERATE NEXT COMMAND JSON", None, False
        elif console_input == "s":
            return self.process_self_feedback(assistant_reply_json, cfg)
        elif console_input == "":
            print("Invalid input format.")
            return user_input, None, True
        elif console_input.startswith("y -"):
            return self.process_continue_for(user_input, console_input)
        elif console_input == "n":
            return "EXIT", None, False
        return console_input, "human_feedback", False

    def log_next_action(self, command_name, arguments):
        send_chat_message_to_user(
            "NEXT ACTION: \n " + f"COMMAND = {command_name} \n "
            f"ARGUMENTS = {arguments}"
        )
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
            f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
        )
        print(
            "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands"
            "'n' to exit program, or enter feedback for "
            f"{self.ai_name}...",
            flush=True,
        )

    def log_auth_or_exit(self, user_input):
        if user_input == "GENERATE NEXT COMMAND JSON":
            logger.typewriter_log(
                "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                Fore.MAGENTA,
                "",
            )
        elif user_input == "EXIT":
            send_chat_message_to_user("Exiting...")
            print("Exiting...", flush=True)
            exit(0)

    def get_user_input(self, command_name, arguments, cfg, assistant_reply_json):
        user_input = self.user_input = ""
        self.log_next_action(command_name, arguments)
        while True:
            console_input = self.get_console_input(cfg).lower().strip()

            user_input, feedback_type, invalid_input = self.process_console_input(
                console_input, cfg, user_input, assistant_reply_json
            )
            if invalid_input:
                continue
            if feedback_type:
                command_name = "human_feedback"
            break
        self.log_auth_or_exit(user_input)
        return user_input

    def log_command(self, command_name, arguments):
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

    def process_command(self, cfg, command_name, arguments):
        for plugin in cfg.plugins:
            if not plugin.can_handle_pre_command():
                continue
            command_name, arguments = plugin.pre_command(command_name, arguments)
        command_result = execute_command(
            self.command_registry,
            command_name,
            arguments,
            self.config.prompt_generator,
        )
        result = f"Command {command_name} returned: " f"{command_result}"

        for plugin in cfg.plugins:
            if not plugin.can_handle_post_command():
                continue
            result = plugin.post_command(command_name, result)
        if self.next_action_count > 0:
            self.next_action_count -= 1
        return result, command_name, arguments

    def execute_command(self, cfg, command_name, arguments, user_input):
        if command_name is not None and command_name.lower().startswith("error"):
            result = f"Command {command_name} threw the following error: {arguments}"
        elif command_name == "human_feedback":
            result = f"Human feedback: {user_input}"
        else:
            result, command_name, arguments = self.process_command(
                cfg, command_name, arguments
            )
        return result, command_name, arguments

    def input_or_continuous(
        self, cfg, command_name, arguments, user_input, assistant_reply_json
    ):
        if not cfg.continuous_mode and self.next_action_count == 0:
            user_input = self.get_user_input(
                command_name, arguments, cfg, assistant_reply_json
            )
            self.user_input = user_input
        else:
            self.log_command(command_name, arguments)
        return user_input

    def interact(self, loop_count, cfg, command_name, arguments, user_input):
        loop_count += 1
        self.check_continuous(cfg, loop_count)
        send_chat_message_to_user("Thinking... \n")
        (
            command_name,
            arguments,
            assistant_reply,
            assistant_reply_json,
        ) = self.process_assistant_reply(cfg)
        user_input = self.input_or_continuous(
            cfg, command_name, arguments, user_input, assistant_reply_json
        )
        result, command_name, arguments = self.execute_command(
            cfg, command_name, arguments, user_input
        )
        self.update_memory(assistant_reply, result, user_input)
        self.update_history(result)
        return loop_count, cfg, command_name, arguments, user_input

    def update_memory(self, assistant_reply, result, user_input):
        self.memory.add(
            (
                f"Assistant Reply: {assistant_reply} "
                f"\nResult: {result} "
                f"\nHuman Feedback: {user_input} "
            )
        )

    def update_history(self, result):
        if result is not None:
            self.full_message_history.append(create_chat_message("system", result))
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            return
        self.full_message_history.append(
            create_chat_message("system", "Unable to execute command")
        )
        logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

    def _resolve_pathlike_command_args(self, command_args):
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
            return command_args
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
            f"Below is a message from an AI agent with the role of {ai_role}."
            "Please review the provided Thought, Reasoning, Plan, and Criticism."
            "If these elements accurately contribute to the successful execution"
            " of the assumed role, respond with the letter 'Y' followed by a space,"
            " and then explain why it is effective. If the provided information is not"
            " suitable for achieving the role's objectives, please provide one or more "
            "sentences addressing the issue and suggesting a resolution."
        )
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")
        thought = thoughts.get("thoughts", "")
        criticism = thoughts.get("criticism", "")
        feedback_thoughts = thought + reasoning + plan + criticism
        return create_chat_completion(
            [{"role": "user", "content": feedback_prompt + feedback_thoughts}],
            llm_model,
        )
