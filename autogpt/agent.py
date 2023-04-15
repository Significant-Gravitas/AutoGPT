import json
import regex
import traceback

from colorama import Fore, Style

from autogpt.chat import chat_with_ai, create_chat_message
import autogpt.commands as cmd
from autogpt.config import Config
from autogpt.json_parser import fix_and_parse_json
from autogpt.logger import logger
from autogpt.speak import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input


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

    def __init__(
        self,
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        prompt,
        user_input,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.prompt = prompt
        self.user_input = user_input

    def start_interaction_loop(self):
        # Interaction Loop
        cfg = Config()
        loop_count = 0
        command_name = None
        arguments = None
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1
            if (
                cfg.continuous_mode
                and cfg.continuous_limit > 0
                and loop_count > cfg.continuous_limit
            ):
                logger.typewriter_log(
                    "Continuous Limit Reached: ", Fore.YELLOW, f"{cfg.continuous_limit}"
                )
                break

            # Send message to AI, get response
            with Spinner("Thinking... "):
                assistant_reply = chat_with_ai(
                    self.prompt,
                    self.user_input,
                    self.full_message_history,
                    self.memory,
                    cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            # Print Assistant thoughts
            print_assistant_thoughts(self.ai_name, assistant_reply)

            # Get command name and arguments
            try:
                command_name, arguments = cmd.get_command(
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
                    "Enter 'y' to authorise command, 'y -N' to run N continuous "
                    "commands, 'n' to exit program, or enter feedback for "
                    f"{self.ai_name}...",
                    flush=True,
                )
                while True:
                    console_input = clean_input(
                        Fore.MAGENTA + "Input:" + Style.RESET_ALL
                    )
                    if console_input.lower().rstrip() == "y":
                        self.user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().startswith("y -"):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            self.user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            print(
                                "Invalid input format. Please enter 'y -n' where n is"
                                " the number of continuous tasks."
                            )
                            continue
                        break
                    elif console_input.lower() == "n":
                        self.user_input = "EXIT"
                        break
                    else:
                        self.user_input = console_input
                        command_name = "human_feedback"
                        break

                if self.user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )
                elif self.user_input == "EXIT":
                    print("Exiting...", flush=True)
                    break
            else:
                # Print command
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
                result = f"Human feedback: {self.user_input}"
            else:
                result = (
                    f"Command {command_name} returned: "
                    f"{cmd.execute_command(command_name, arguments)}"
                )
                if self.next_action_count > 0:
                    self.next_action_count -= 1

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} "
                f"\nResult: {result} "
                f"\nHuman Feedback: {self.user_input} "
            )

            self.memory.add(memory_to_add)

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

    # All other errors, return "Error: + error message"
    except Exception:
        call_stack = traceback.format_exc()
        logger.error("Error: \n", call_stack)
