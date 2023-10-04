"""Logging module for Auto-GPT."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from colorama import Fore

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.singleton import Singleton

from .formatters import AutoGptFormatter, JsonFormatter
from .handlers import ConsoleHandler, JsonFileHandler, TypingConsoleHandler


class Logger(metaclass=Singleton):
    """
    Logger that handle titles in different colors.
    Outputs logs in console, activity.log, and errors.log
    For console handler: simulates typing
    """

    def __init__(self):
        # create log directory if it doesn't exist
        # TODO: use workdir from config
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        if not self.log_dir.exists():
            self.log_dir.mkdir()

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
        self.file_handler = logging.FileHandler(self.log_dir / log_file, "a", "utf-8")
        self.file_handler.setLevel(logging.DEBUG)
        info_formatter = AutoGptFormatter(
            "%(asctime)s %(levelname)s %(title)s %(message_no_color)s"
        )
        self.file_handler.setFormatter(info_formatter)

        # Error handler error.log
        error_handler = logging.FileHandler(self.log_dir / error_file, "a", "utf-8")
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

        self.json_logger = logging.getLogger("JSON_LOGGER")
        self.json_logger.addHandler(self.file_handler)
        self.json_logger.addHandler(error_handler)
        self.json_logger.setLevel(logging.DEBUG)

        self._config: Optional[Config] = None
        self.chat_plugins = []

    @property
    def config(self) -> Config | None:
        return self._config

    @config.setter
    def config(self, config: Config):
        self._config = config
        if config.plain_output:
            self.typing_logger.removeHandler(self.typing_console_handler)
            self.typing_logger.addHandler(self.console_handler)

    def typewriter_log(
        self,
        title: str = "",
        title_color: str = "",
        content: str = "",
        speak_text: bool = False,
        level: int = logging.INFO,
    ) -> None:
        from autogpt.speech import say_text

        if speak_text and self.config and self.config.speak_mode:
            say_text(f"{title}. {content}", self.config)

        for plugin in self.chat_plugins:
            plugin.report(f"{title}. {content}")

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
        message: str,
        title: str = "",
        title_color: str = "",
    ) -> None:
        self._log(title, title_color, message, logging.DEBUG)

    def info(
        self,
        message: str,
        title: str = "",
        title_color: str = "",
    ) -> None:
        self._log(title, title_color, message, logging.INFO)

    def warn(
        self,
        message: str,
        title: str = "",
        title_color: str = "",
    ) -> None:
        self._log(title, title_color, message, logging.WARN)

    def error(self, title: str, message: str = "") -> None:
        self._log(title, Fore.RED, message, logging.ERROR)

    def _log(
        self,
        title: str = "",
        title_color: str = "",
        message: str = "",
        level: int = logging.INFO,
    ) -> None:
        if message:
            if isinstance(message, list):
                message = " ".join(message)
        self.logger.log(
            level, message, extra={"title": str(title), "color": str(title_color)}
        )

    def set_level(self, level: logging._Level) -> None:
        self.logger.setLevel(level)
        self.typing_logger.setLevel(level)

    def double_check(self, additionalText: Optional[str] = None) -> None:
        if not additionalText:
            additionalText = (
                "Please ensure you've setup and configured everything"
                " correctly. Read https://github.com/Torantulino/Auto-GPT#readme to "
                "double check. You can also create a github issue or join the discord"
                " and ask there!"
            )

        self.typewriter_log("DOUBLE CHECK CONFIGURATION", Fore.YELLOW, additionalText)

    def log_json(self, data: Any, file_name: str | Path) -> None:
        # Create a handler for JSON files
        json_file_path = self.log_dir / file_name
        json_data_handler = JsonFileHandler(json_file_path)
        json_data_handler.setFormatter(JsonFormatter())

        # Log the JSON data using the custom file handler
        self.json_logger.addHandler(json_data_handler)
        self.json_logger.debug(data)
        self.json_logger.removeHandler(json_data_handler)


logger = Logger()
