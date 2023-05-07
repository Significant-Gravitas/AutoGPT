from datetime import datetime
from typing import Callable, Dict

from autogpt.core.messaging.base import (
    Role,
    Sender,
    MessageMetadata,
    Message,
    MessageChannel,
    MessageBroker,
)


class SimpleMessageChannel(MessageChannel):

    def __init__(
        self,
        sender_name: str,
        sender_role: Role,
        message_broker: "MessageBroker",
    ):
        self._sender = Sender(name=sender_name, role=sender_role)
        self._message_broker = message_broker
        self._listeners = []

    def add_listener(
        self,
        listener: Callable[[Message], None],
        message_filter: Callable[[Message], Message | None] = lambda x: x,
    ) -> None:
        self._listeners.append((listener, message_filter))

    def send_message(self, content: Dict) -> None:
        metadata = MessageMetadata(self._sender, datetime.now())
        message = Message(
            content=content,
            metadata=metadata,
        )
        for listener, message_filter in self._listeners:
            if filtered_message := message_filter(message):
                listener(filtered_message)

    def __repr__(self) -> str:
        listeners = [listener.__name__ for listener, _ in self._listeners]
        return f"SimpleMessageChannel(sender={self._sender}, listeners={listeners})"


class SimpleMessageBroker(MessageBroker):

    def __init__(self, logger):
        self._logger = logger
        self._channels: Dict[str, SimpleMessageChannel] = {}

    def get_message_channel(
        self,
        sender_name: str,
        sender_role: Role,
    ) -> SimpleMessageChannel:

        if sender_name not in self._channels:
            self._channels[sender_name] = SimpleMessageChannel(
                sender_name,
                sender_role,
                self,
            )
        return self._channels[sender_name]

    def send_message(self, sender_name: str, sender_role: Role, content: Dict) -> None:
        self.get_message_channel(sender_name, sender_role).send_message(content)

    def register_listener(
        self,
        message_channel: str,
        listener_callback: Callable[[Message], None],
    ) -> None:
        self._channels[message_channel].add_listener(listener_callback)

