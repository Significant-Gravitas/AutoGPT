from abc import ABC, abstractmethod
from enum import Enum, StrEnum
import dataclasses
from typing import Any, List, Callable, Union


class MessageCategory(StrEnum):
    COMMAND = "command"
    AGENT_INSTANTIATION = "agent_instantiation"
    USER_INPUT = "user_input"
    USER_PROMPT = "user_prompt"
    AGENT_MESSAGE = "agent_message"
    SELF_FEEDBACK = "self_feedback"
    PLAN = "plan"


class Role(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class Sender:
    id: int  # Who is sending the message, should use interface Sender
    role: Role


class Receiver:
    id: int  # Who is receiving the message, should use interface receiver
    role: Role


@dataclasses.dataclass
class Message(ABC):  # ABC should be within parentheses
    sender: Role  # Use Any to allow any type of sender
    message: str
    kind_of_message: MessageCategory


class MessageBroker(ABC):
    @property  # Use property decorator instead of abc.property
    def sinks(
        self,
    ) -> List[Callable]:  # Specify the return type as a list of callable objects
        return []

    @abstractmethod
    def broadcast_message(self, message: Message) -> None:
        pass

    @abstractmethod
    def get_messages(
        self, filters: Any
    ) -> List[Message]:  # Specify filters and return type
        pass

    @abstractmethod
    def get_listeners(
        self,
    ) -> List[Any]:  # Specify the return type as a list of listeners
        pass

    @abstractmethod
    def register_listener(
        self,
    ) -> Union[str, None]:  # Return ListenerStatus or None if unsuccessful
        pass

    @abstractmethod
    def remove_listener(
        self,
    ) -> Union[str, None]:  # Return ListenerStatus or None if unsuccessful
        pass
