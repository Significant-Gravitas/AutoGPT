"""Base class for interaction providers."""
import abc
import dataclasses

from autogpt.logs import Logger
from autogpt.singleton import AbstractSingleton


@dataclasses.dataclass
class OutputMessage:
    """A dataclass for output messages"""

    title: str
    title_color: str
    content: str
    speak_text: bool
    level: int
    context: dict


@dataclasses.dataclass
class InputMessage:
    """A dataclass for input messages"""

    title: str
    title_color: str
    request_content: str
    level: int
    message: str
    context: dict


class InteractionProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def __init__(self, logger: Logger):
        """Inits the provider"""
        pass

    @property
    @abc.abstractmethod
    def logger(self):
        """Returns the logger instance"""
        pass

    @abc.abstractmethod
    def input(self, message: InputMessage):
        """Gets input from input-er"""
        pass

    @abc.abstractmethod
    def output(self, message: OutputMessage):
        """Gets the output-er
        This needs more data passed in similiar to the typewriter_log
        """
