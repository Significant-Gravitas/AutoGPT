import json
import logging
import random
import time
from pathlib import Path


class ConsoleHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            print(msg)
        except Exception:
            self.handleError(record)


class TypingConsoleHandler(logging.StreamHandler):
    """Output stream to console using simulated typing"""

    def emit(self, record: logging.LogRecord):
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


class JsonFileHandler(logging.FileHandler):
    def __init__(self, filename: str | Path, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record: logging.LogRecord):
        json_data = json.loads(self.format(record))
        with open(self.baseFilename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
