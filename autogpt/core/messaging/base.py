"""Abstract interface for a message broker system."""
import abc
import enum
from datetime import datetime
from typing import Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel


class Role(str, enum.Enum):
    # Client side
    USER = "user"
    CLIENT_APPLICATION = "client_application"
    # Server side
    APPLICATION_SERVER = "application_server"
    AGENT_FACTORY = "agent_factory"
    # Agent side
    AGENT = "agent"


class Sender(BaseModel):
    """Struct for metadata about a sender."""

    id: UUID
    name: str
    role: Role


class MessageContent(BaseModel):
    pass


class Message(BaseModel):
    """Struct for a message and its metadata."""

    sender: Sender
    timestamp: datetime
    additional_metadata: dict = None
    content: MessageContent


Listener = Callable[[Message], Awaitable[None]]
MessageFilter = Callable[[Message], bool]


class MessageEmitter(abc.ABC):
    """Interface for an emitter that can send messages on a particular channel.

    Emitters are used to send messages on a particular channel. They preserve the
    metadata associated with them when they are created so client code can simply
    pass messages to them without having to worry about the details of the channel
    or repeatedly passing metadata.

    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def send_message(self, content: MessageContent, **extra_metadata) -> bool:
        """Send a message on this channel.

        Args:
            content: A json-serializable object containing the message content.
            **extra_metadata: Any additional metadata to be included when emitting messages.

        Returns:
            True if the message was sent successfully, False otherwise.

        """
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...


class MessageChannel(abc.ABC):
    """Interface for a channel that can propagate messages to a group of listeners."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def add_listener(
        self,
        listener: Listener,
        message_filter: MessageFilter,
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
    def send_message(self, message: Message) -> bool:
        """Send a message on this channel.

        Args:
            message: The message to send.

        Returns:
            True if the message was sent successfully, False otherwise.

        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        ...


class MessageBroker(abc.ABC):
    """Interface for a broker that can create message channels and propagate messages."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def create_message_channel(
        self,
        channel_name: str,
    ) -> None:
        """Construct a message channel with a given name.

        Args:
            channel_name: The name of the channel to create.

        """
        ...

    @abc.abstractmethod
    def get_emitter(
        self,
        channel_name: str,
        sender_uuid: UUID,
        sender_name: str,
        sender_role: Role,
        **extra_metadata,
    ) -> MessageEmitter:
        """Get an emitter for this channel.

        Args:
            channel_name: The name of the channel the emitter will send messages on.
            sender_uuid: The UUID of the sender.
            sender_name: The name of the sender.
            sender_role: The role of the sender.
            extra_metadata: Any additional metadata to be included when emitting messages.

        Returns:
            A MessageEmitter object that can be used to send messages on this channel
            preserving all metadata.

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
