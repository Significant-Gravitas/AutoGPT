from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import TYPE_CHECKING

from autogpt.logs.utils import remove_color_codes
from autogpt.speech import TextToSpeechProvider

if TYPE_CHECKING:
    from autogpt.speech import TTSConfig


class TypingConsoleHandler(logging.StreamHandler):
    """Output stream to console using simulated typing"""

    # Typing speed settings in WPS (Words Per Second)
    MIN_WPS = 25
    MAX_WPS = 100

    def emit(self, record: logging.LogRecord) -> None:
        min_typing_interval = 1 / TypingConsoleHandler.MAX_WPS
        max_typing_interval = 1 / TypingConsoleHandler.MIN_WPS

        msg = self.format(record)
        try:
            # Split without discarding whitespace
            words = re.findall(r"\S+\s*", msg)

            for i, word in enumerate(words):
                self.stream.write(word)
                self.flush()
                if i >= len(words) - 1:
                    self.stream.write(self.terminator)
                    self.flush()
                    break

                interval = random.uniform(min_typing_interval, max_typing_interval)
                # type faster after each word
                min_typing_interval = min_typing_interval * 0.95
                max_typing_interval = max_typing_interval * 0.95
                time.sleep(interval)
        except Exception:
            self.handleError(record)


class TTSHandler(logging.Handler):
    """Output messages to the configured TTS engine (if any)"""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config
        self.tts_provider = TextToSpeechProvider(config)

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "title", ""):
            msg = f"{getattr(record, 'title')} {record.msg}"
        else:
            msg = f"{record.msg}"

        return remove_color_codes(msg)

    def emit(self, record: logging.LogRecord) -> None:
        if not self.config.speak_mode:
            return

        message = self.format(record)
        self.tts_provider.say(message)


class JsonFileHandler(logging.FileHandler):
    def format(self, record: logging.LogRecord) -> str:
        record.json_data = json.loads(record.getMessage())
        return json.dumps(getattr(record, "json_data"), ensure_ascii=False, indent=4)

    def emit(self, record: logging.LogRecord) -> None:
        with open(self.baseFilename, "w", encoding="utf-8") as f:
            f.write(self.format(record))
