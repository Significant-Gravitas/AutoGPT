import logging
import os
import random
import time
from logging import LogRecord

from colorama import Style

import speak
from config import Config
from config import Singleton

import re

cfg = Config()

'''
Logger that handle titles in different colors.
Outputs logs in console, activity.log, and errors.log
For console handler: simulates typing
'''


class Logger(metaclass=Singleton):
    def __init__(self):
        # create log directory if it doesn't exist
        log_dir = os.path.join('..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = "activity.log"
        error_file = "error.log"

        # Create a handler for INFO level logs
        self.console_handler = TypingConsoleHandler()
        self.console_handler.setLevel(logging.INFO)
        console_formatter = AutoGptFormatter('%(title_color)s %(message)s')
        self.console_handler.setFormatter(console_formatter)

        # Info handler in activity.log
        self.file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        self.file_handler.setLevel(logging.INFO)
        info_formatter = AutoGptFormatter('%(asctime)s %(levelname)s %(title)s %(message_no_color)s')
        self.file_handler.setFormatter(info_formatter)

        # Error handler error.log
        error_handler = logging.FileHandler(os.path.join(log_dir, error_file))
        error_handler.setLevel(logging.ERROR)
        error_formatter = AutoGptFormatter(
            '%(asctime)s %(levelname)s %(module)s:%(funcName)s:%(lineno)d %(title)s %(message_no_color)s')
        error_handler.setFormatter(error_formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(error_handler)

    def log(
            self,
            title='',
            title_color='',
            content='',
            speak_text=False,
            level=logging.INFO):
        if speak_text and cfg.speak_mode:
            speak.say_text(f"{title}. {content}")

        if content:
            if isinstance(content, list):
                content = " ".join(content)
        else:
            content = ""

        self.logger.log(level, content, extra={'title': title, 'color': title_color})

    def set_level(self, level):
        self.logger.setLevel(level)
        self.console_handler.setLevel(level)
        self.file_handler.setLevel(level)


'''
Output stream to console using simulated typing
'''


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


'''
Allows to handle custom placeholders 'title_color' and 'message_no_color'.
To use this formatter, make sure to pass 'color', 'title' as log extras.
'''


class AutoGptFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        record.title_color = getattr(record, 'color') + getattr(record, 'title') + " " + Style.RESET_ALL
        if hasattr(record, 'msg'):
            record.message_no_color = remove_color_codes(getattr(record, 'msg'))
        else:
            record.message_no_color = ''
        return super().format(record)


def remove_color_codes(s: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s)


logger = Logger()
