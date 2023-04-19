"""Logging module for Auto-GPT."""
import json
import logging
import os
import random
import re
import time
import traceback
from logging import LogRecord

from colorama import Fore, Style

from autogpt.config import Config, Singleton
from autogpt.speech import say_text

CFG = Config()


class Logger(metaclass=Singleton):
    """
    Logger that handle titles in different colors.
    Outputs logs in console, activity.log, and errors.log
    For console handler: simulates typing
    """

    def __init__(self):
        # create log directory if it doesn't exist
        this_files_dir_path = os.path.dirname(__file__)
        log_dir = os.path.join(this_files_dir_path, "../logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = "activity.log"
        error_file = "error.log"

        console_formatter = AutoGptFormatter("%(title_color)s %(message)s")

        # Create a handler for console which simulate typing
        self.typing_console_handler = TypingConsoleHandler()
        self.typing_console_handler.setLevel(logging.INFO)
        self.typing_console_handler.setFormatter(console_formatter)

        # Create a handler for console without typing simulation
        self.console_handler = ConsoleHandler()
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.setFormatter(console_formatter)

        # Info handler in activity.log
        self.file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file), "a", "utf-8"
        )
        self.file_handler.setLevel(logging.DEBUG)
        info_formatter = AutoGptFormatter(
            "%(asctime)s %(levelname)s %(title)s %(message_no_color)s"
        )
        self.file_handler.setFormatter(info_formatter)

        # Error handler error.log
        error_handler = logging.FileHandler(
            os.path.join(log_dir, error_file), "a", "utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = AutoGptFormatter(
            "%(asctime)s %(levelname)s %(module)s:%(funcName)s:%(lineno)d %(title)s"
            " %(message_no_color)s"
        )
        error_handler.setFormatter(error_formatter)

        self.typing_logger = logging.getLogger("TYPER")
        self.typing_logger.addHandler(self.typing_console_handler)
        self.typing_logger.addHandler(self.file_handler)
        self.typing_logger.addHandler(error_handler)
        self.typing_logger.setLevel(logging.DEBUG)

        self.logger = logging.getLogger("LOGGER")
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.DEBUG)

    def typewriter_log(
        self, title="", title_color="", content="", speak_text=False, level=logging.INFO
    ):
        if speak_text and CFG.speak_mode:
            say_text(f"{title}. {content}")

        if content:
            if isinstance(content, list):
                content = " ".join(content)
        else:
            content = ""

        self.typing_logger.log(
            level, content, extra={"title": title, "color": title_color}
        )

    def debug(
        self,
        message,
        title="",
        title_color="",
    ):
        self._log(title, title_color, message, logging.DEBUG)

    def warn(
        self,
        message,
        title="",
        title_color="",
    ):
        self._log(title, title_color, message, logging.WARN)

    def error(self, title, message=""):
        self._log(title, Fore.RED, message, logging.ERROR)

    def _log(self, title="", title_color="", message="", level=logging.INFO):
        if message:
            if isinstance(message, list):
                message = " ".join(message)
        self.logger.log(level, message, extra={"title": title, "color": title_color})

    def set_level(self, level):
        self.logger.setLevel(level)
        self.typing_logger.setLevel(level)

    def double_check(self, additionalText=None):
        if not additionalText:
            additionalText = (
                "Please ensure you've setup and configured everything"
                " correctly. Read https://github.com/Torantulino/Auto-GPT#readme to "
                "double check. You can also create a github issue or join the discord"
                " and ask there!"
            )

        self.typewriter_log("DOUBLE CHECK CONFIGURATION", Fore.YELLOW, additionalText)


"""
Output stream to console using simulated typing
"""


class TypingConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        min_typing_speed = 0.05
        max_typing_speed = 0.01

        msg = self.format(record)
        try:
            words = msg.split()
            for i, word in enumerate(words):
                print(word, end="", flush=True)
                if i < len(words) - 1:
                    print(" ", end="", flush=True)
                typing_speed = random.uniform(min_typing_speed, max_typing_speed)
                time.sleep(typing_speed)
                # type faster after each word
                min_typing_speed = min_typing_speed * 0.95
                max_typing_speed = max_typing_speed * 0.95
            print()
        except Exception:
            self.handleError(record)


class ConsoleHandler(logging.StreamHandler):
    def emit(self, record) -> None:
        msg = self.format(record)
        try:
            print(msg)
        except Exception:
            self.handleError(record)


class AutoGptFormatter(logging.Formatter):
    """
    Allows to handle custom placeholders 'title_color' and 'message_no_color'.
    To use this formatter, make sure to pass 'color', 'title' as log extras.
    """

    def format(self, record: LogRecord) -> str:
        if hasattr(record, "color"):
            record.title_color = (
                getattr(record, "color")
                + getattr(record, "title")
                + " "
                + Style.RESET_ALL
            )
        else:
            record.title_color = getattr(record, "title")
        if hasattr(record, "msg"):
            record.message_no_color = remove_color_codes(getattr(record, "msg"))
        else:
            record.message_no_color = ""
        return super().format(record)


def remove_color_codes(s: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", s)


logger = Logger()


def print_assistant_thoughts(ai_name, assistant_reply):
    """Prints the assistant's thoughts to the console"""
    from autogpt.json_utils.json_fix_llm import (
        attempt_to_fix_json_by_finding_outermost_brackets,
        fix_and_parse_json,
    )

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
        if CFG.speak_mode and assistant_thoughts_speak:
            say_text(assistant_thoughts_speak)
        else:
            logger.typewriter_log("SPEAK:", Fore.YELLOW, f"{assistant_thoughts_speak}")

        return assistant_reply_json
    except json.decoder.JSONDecodeError:
        logger.error("Error: Invalid JSON\n", assistant_reply)
        if CFG.speak_mode:
            say_text(
                "I have received an invalid JSON response from the OpenAI API."
                " I cannot ignore this response."
            )

    # All other errors, return "Error: + error message"
    except Exception:
        call_stack = traceback.format_exc()
        logger.error("Error: \n", call_stack)


def print_assistant_thoughts(
    ai_name: object, assistant_reply_json_valid: object
) -> None:
    assistant_thoughts_reasoning = None
    assistant_thoughts_plan = None
    assistant_thoughts_speak = None
    assistant_thoughts_criticism = None

    assistant_thoughts = assistant_reply_json_valid.get("thoughts", {})
    assistant_thoughts_text = assistant_thoughts.get("text")
    if assistant_thoughts:
        assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
        assistant_thoughts_plan = assistant_thoughts.get("plan")
        assistant_thoughts_criticism = assistant_thoughts.get("criticism")
        assistant_thoughts_speak = assistant_thoughts.get("speak")
    logger.typewriter_log(
        f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, f"{assistant_thoughts_text}"
    )
    logger.typewriter_log("REASONING:", Fore.YELLOW, f"{assistant_thoughts_reasoning}")
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
    logger.typewriter_log("CRITICISM:", Fore.YELLOW, f"{assistant_thoughts_criticism}")
    # Speak the assistant's thoughts
    if CFG.speak_mode and assistant_thoughts_speak:
        say_text(assistant_thoughts_speak)
