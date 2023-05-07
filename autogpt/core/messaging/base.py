from abc import ABC, abstractmethod
import dataclasses
from typing import Callable


@dataclasses.dataclass
class Message(ABC):  # ABC should be within parentheses
    message_channel: str
    content: dict  # Some json struct we can define with a strict schema


class MessageBroker(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def register_listener(
        self,
        message_channel: str,
        listener_callback: Callable[[Message], None],
    ) -> None:
        pass

    @abstractmethod
    def send_message(self, message_channel: str, message: Message) -> None:
        pass







