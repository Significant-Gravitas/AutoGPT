import json

from colorama import Fore, Style
from regex import regex

from autogpt import chat, utils
from autogpt.app import execute_command, get_command
from autogpt.chat import cfg
from autogpt.config import Config
from autogpt.json_fixes.bracket_termination import (
    attempt_to_fix_json_by_finding_outermost_brackets,
)
from autogpt.json_fixes.parsing import fix_and_parse_json
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
        next_action_count: The number of actions to execute.
        prompt: The prompt to use.
        user_input: The user input.

    """

    def __init__(self,
                 ai_name,
                 memory,
                 full_message_history,
                 next_action_count,
                 prompt,
                 user_input):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.prompt = prompt
        self.user_input = user_input
        self.assistant_reply = None
        self.command_name = ""
        self.command_arguments = None
        self.command_result = None
        self.exit_loop = False
        self.loop_count = 0

    def start_interaction_loop(self):

        self.loop_count = 0
        while not self.exit_loop:

            self.loop_count += 1

            response = self.start_agent()

            if response == 'continue':
                continue

            self.check_limit_loop_count()

    def start_agent(self):
        self.send_message_to_ai()

        print_assistant_thoughts(self.ai_name, self.assistant_reply)

        self.get_command_name_and_args()

        if cfg.continuous_mode or self.next_action_count > 0:
            self.go_to_next_command()
            return "continue"

        self.handle_user_input()

        if not self.exit_loop:
            self.execute_command()

            self.add_data_to_memory()

            self.add_message_to_full_message_history()

    def handle_user_input(self):

        self.user_input = ""
        self.print_next_command_action()
        print(
            f"Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for {self.ai_name}...",
            flush=True)

        not_a_good_input = True
        while not_a_good_input:
            not_a_good_input = self.check_valid_input()

        if self.user_input == "GENERATE NEXT COMMAND JSON":
            logger.typewriter_log(
                "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                Fore.MAGENTA,
                "")
        elif self.user_input == "EXIT":
            print("Exiting...", flush=True)
            self.exit_loop = True

    def get_command_name_and_args(self):
        try:
            self.command_name, self.command_arguments = get_command(
                attempt_to_fix_json_by_finding_outermost_brackets(self.assistant_reply))
            if cfg.speak_mode:
                cfg.speak.say_text(f"I want to execute {self.command_name}")
        except Exception as e:
            logger.error("Error: \n", str(e))

    def execute_command(self):
        # Execute command
        if self.command_name is not None and self.command_name.lower().startswith("error"):
            result = f"Command {self.command_name} threw the following error: " + self.command_arguments
        elif self.command_name == "human_feedback":
            result = f"Human feedback: {self.user_input}"
        else:
            result = f"Command {self.command_name} returned: " \
                     f"{execute_command(self.command_name, self.command_arguments)}"
            if self.next_action_count > 0:
                self.next_action_count -= 1

        return result

    def add_data_to_memory(self):
        memory_to_add = f"Assistant Reply: {self.assistant_reply} " \
                        f"\nResult: {self.command_result} " \
                        f"\nHuman Feedback: {self.user_input} "

        self.memory.add(memory_to_add)

    def add_message_to_full_message_history(self):
        if self.command_result is not None:
            self.full_message_history.append(chat.create_chat_message("system", self.command_result))
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, self.command_result)
        else:
            self.full_message_history.append(
                chat.create_chat_message(
                    "system", "Unable to execute command"))
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

    def check_valid_input(self):
        GENERATE_NEXT_COMMAND = "GENERATE NEXT COMMAND JSON"

        console_input = utils.clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)

        if console_input.lower().rstrip() == "y":
            self.user_input = GENERATE_NEXT_COMMAND
            return False

        if console_input.lower().startswith("y -"):
            try:
                self.next_action_count = abs(int(console_input.split(" ")[1]))
                self.user_input = GENERATE_NEXT_COMMAND
            except ValueError:
                self.command_name, self.args_name = get_command(
                    attempt_to_fix_json_by_finding_outermost_brackets(self.assistant_reply)
                )
                if cfg.speak_mode:
                    say_text(f"I want to execute {self.command_name}")
            except Exception as e:
                logger.error("Error: \n", str(e))

            if not cfg.continuous_mode and self.next_action_count == 0:
                ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{self.command_name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{self.command_arguments}{Style.RESET_ALL}",
                )
                print(
                    "Invalid input format. Please enter 'y -n' where n is the number of continuous tasks.")
                return True
            return False

        if console_input.lower() == "n":
            self.user_input = "EXIT"
            return False

        self.user_input = console_input
        self.command_name = "human_feedback"
        return False

    def send_message_to_ai(self):
        # Send message to AI, get response
        with Spinner("Thinking... "):
            self.assistant_reply = chat.chat_with_ai(
                self.prompt,
                self.user_input,
                self.full_message_history,
                self.memory,
                cfg.fast_token_limit)  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

    def go_to_next_command(self):
        if self.next_action_count > 0:
            self.next_action_count -= 1
        self.print_next_command_action()

    def print_next_command_action(self):
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{self.command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}"
            f"{self.command_arguments}{Style.RESET_ALL}")

    def check_limit_loop_count(self):
        if self.next_action_count < 0:
            self.exit_loop = True

        if cfg.continuous_mode and self.loop_count > cfg.continuous_limit > 0:
            logger.typewriter_log("Continuous Limit Reached: ", Fore.YELLOW, f"{cfg.continuous_limit}")
            self.exit_loop = True
