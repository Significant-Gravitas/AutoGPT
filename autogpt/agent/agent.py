from colorama import Fore, Style
from autogpt.app import execute_command, get_command

from autogpt import chat, utils, speak
from autogpt.chat import cfg
import autogpt.commands as cmd
from autogpt.chat import chat_with_ai, create_chat_message
from autogpt.config import Config
from autogpt.json_fixes.bracket_termination import (
    attempt_to_fix_json_by_finding_outermost_brackets,
)
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
        self.command_name = None
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
            self.command_name, self.command_arguments = cmd.get_command(
                attempt_to_fix_json_by_finding_outermost_brackets(self.assistant_reply))
            if cfg.speak_mode:
                speak.say_text(f"I want to execute {self.command_name}")
        except Exception as e:
            logger.error("Error: \n", str(e))

    def execute_command(self):
        # Execute command
        result = None
        if self.command_name is not None and self.command_name.lower().startswith("error"):
            result = f"Command {self.command_name} threw the following error: " + self.command_arguments
        elif self.command_name == "human_feedback":
            result = f"Human feedback: {self.user_input}"
        else:
            result = f"Command {self.command_name} returned: " \
                     f"{cmd.execute_command(self.command_name, self.command_arguments)}"
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
                command_name, arguments = get_command(
                    attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply)
                )
                if cfg.speak_mode:
                    say_text(f"I want to execute {command_name}")
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
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
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


def attempt_to_fix_json_by_finding_outermost_brackets(json_string):
    cfg = Config()
    if cfg.speak_mode and cfg.debug_mode:
        say_text(
            "I have received an invalid JSON response from the OpenAI API. "
            "Trying to fix it now."
        )
    logger.typewriter_log("Attempting to fix JSON by finding outermost brackets\n")

    try:
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        json_match = json_pattern.search(json_string)

        if json_match:
            # Extract the valid JSON object from the string
            json_string = json_match.group(0)
            logger.typewriter_log(
                title="Apparently json was fixed.", title_color=Fore.GREEN
            )
            if cfg.speak_mode and cfg.debug_mode:
                say_text("Apparently json was fixed.")
        else:
            raise ValueError("No valid JSON object found")

    except (json.JSONDecodeError, ValueError):
        if cfg.speak_mode:
            say_text("Didn't work. I will have to ignore this response then.")
        logger.error("Error: Invalid JSON, setting it to empty JSON now.\n")
        json_string = {}

    return json_string


def print_assistant_thoughts(ai_name, assistant_reply):
    """Prints the assistant's thoughts to the console"""
    cfg = Config()
    try:
        try:
            # Parse and print Assistant response
            assistant_reply_json = fix_and_parse_json(assistant_reply)
        except json.JSONDecodeError:
            logger.error("Error: Invalid JSON in assistant thoughts\n", assistant_reply)
            assistant_reply_json = attempt_to_fix_json_by_finding_outermost_brackets(
                assistant_reply
            )
            if isinstance(assistant_reply_json, str):
                assistant_reply_json = fix_and_parse_json(assistant_reply_json)

        # Check if assistant_reply_json is a string and attempt to parse
        # it into a JSON object
        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = json.loads(assistant_reply_json)
            except json.JSONDecodeError:
                logger.error("Error: Invalid JSON\n", assistant_reply)
                assistant_reply_json = (
                    attempt_to_fix_json_by_finding_outermost_brackets(
                        assistant_reply_json
                    )
                )

        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None
        if not isinstance(assistant_reply_json, dict):
            assistant_reply_json = {}
        assistant_thoughts = assistant_reply_json.get("thoughts", {})
        assistant_thoughts_text = assistant_thoughts.get("text")

        if assistant_thoughts:
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
            assistant_thoughts_speak = assistant_thoughts.get("speak")

        logger.typewriter_log(
            f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, f"{assistant_thoughts_text}"
        )
        logger.typewriter_log(
            "REASONING:", Fore.YELLOW, f"{assistant_thoughts_reasoning}"
        )

        if assistant_thoughts_plan:
            logger.typewriter_log("PLAN:", Fore.YELLOW, "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split("\n")
            for line in lines:
                line = line.lstrip("- ")
                logger.typewriter_log("- ", Fore.GREEN, line.strip())

        logger.typewriter_log(
            "CRITICISM:", Fore.YELLOW, f"{assistant_thoughts_criticism}"
        )
        # Speak the assistant's thoughts
        if cfg.speak_mode and assistant_thoughts_speak:
            say_text(assistant_thoughts_speak)

        return assistant_reply_json
    except json.decoder.JSONDecodeError:
        logger.error("Error: Invalid JSON\n", assistant_reply)
        if cfg.speak_mode:
            say_text(
                "I have received an invalid JSON response from the OpenAI API."
                " I cannot ignore this response."
            )
