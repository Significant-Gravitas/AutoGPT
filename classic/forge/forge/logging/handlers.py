from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from forge.logging.utils import remove_color_codes
from forge.speech import TextToSpeechProvider

if TYPE_CHECKING:
    from forge.speech import TTSConfig


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
