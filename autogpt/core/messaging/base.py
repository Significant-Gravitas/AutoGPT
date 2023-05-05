from abc import ABC, abstractmethod
from enum import Enum
import dataclasses
from typing import Any, List, Callable, Union

class MessageCategory(Enum):
    USER_INPUT = 0
    USER_PROMPT = 1

@dataclasses.dataclass
class Message(ABC):  # ABC should be within parentheses
    sender: Any  # Use Any to allow any type of sender
    message: str
    kind_of_message: MessageCategory

class MessageBroker(ABC):

    @property  # Use property decorator instead of abc.property
    def sinks(self) -> List[Callable]:  # Specify the return type as a list of callable objects
        return []

    @abstractmethod
    def broadcast_message(self, message: Message) -> None:
        pass

    @abstractmethod
    def get_messages(self, filters: Any) -> List[Message]:  # Specify filters and return type
        pass

    @abstractmethod
    def get_listeners(self) -> List[Any]:  # Specify the return type as a list of listeners
        pass

    @abstractmethod
    def register_listener(self) -> Union[str, None]:  # Return ListenerStatus or None if unsuccessful
        pass

    @abstractmethod
    def remove_listener(self) -> Union[str, None]:  # Return ListenerStatus or None if unsuccessful
        pass
