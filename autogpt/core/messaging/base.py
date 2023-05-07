"""Abstract interface for a message broker system."""
import abc
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Callable, Dict


class Role(StrEnum):
    """Enumeration of all entities capable of sending messages"""

    USER = "user"
    CLIENT_APPLICATION = "client_application"
    AGENT_FACTORY = "agent_factory"
    AGENT = "agent"


@dataclass
class Sender:
    """Struct for metadata about a sender."""

    # We'll need some kind of uuids at some point.  Names are fine for now.
    name: str
    role: Role


@dataclass
class MessageMetadata:
    """Struct for metadata about a message."""

    sender: Sender
    timestamp: datetime
    # TODO: what else goes here?


@dataclass
class Message:
    """Struct for a message and its metadata."""

    content: dict  # Some json struct we can define with a strict schema
    metadata: MessageMetadata


Listener = Callable[[Message], None]
MessageFilter = Callable[[Message], bool]


class MessageChannel:
    """Interface for a channel that can propagate messages to a group of listeners."""

    @abc.abstractmethod
    def __init__(
        self,
        sender: str,
        role: Role,
        message_broker: "MessageBroker",
    ):
        ...

    @abc.abstractmethod
    def add_listener(
        self,
        listener: Listener,
        message_filter: MessageFilter = lambda x: True,
    ) -> None:
        """Add a listener to this channel.

        Args:
            listener: A callback function that responds to messages
                        sent on this channel.
            message_filter: A function that filters messages before they are sent to
                              the listener.  If the filter returns False, the message
                              is not sent to the listener.

        """
        ...

    @abc.abstractmethod
    def send_message(self, content: dict) -> None:
        """Send a message on this channel.

        Args:
            content: A json-serializable object containing the message content.

        """
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...


class MessageBroker(abc.ABC):
    """Interface for a broker that can create message channels and propagate messages."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def get_message_channel(
        self,
        sender_name: str,
        sender_role: Role,
    ) -> "MessageChannel":
        """Get a message channel for a given sender.

        Args:
            sender_name: A unique name for the sender.
            sender_role: The role of the sender. See the Role enum for options.

        Returns:
            A MessageChannel object the sender can use to propagate messages.

        """
        ...

    def send_message(self, sender_name: str, sender_role: Role, content: Dict) -> None:
        """Send a message on a channel for a given sender.

        This is an alternative method for sending messages by interacting with the
        message broker directly.  It is equivalent to calling `send_message` with an
        appropriate message channel.

        Args:
            sender_name: A unique name for the sender.
            sender_role: The role of the sender. See the Role enum for options.
            content: A json-serializable object containing the message content.

        """
        ...

    @abc.abstractmethod
    def register_listener(
        self,
        message_channel: str,
        listener: Listener,
        message_filter: MessageFilter,
    ) -> None:
        """Register a listener for a given message channel.

        Args:
            message_channel: The name of the message channel to listen to.
            listener: A callback function that responds to messages
                        sent on this channel.
            message_filter: A function that filters messages before they are sent to
                              the listener.  If the filter returns False, the message

        """
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...
