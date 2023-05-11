from datetime import datetime
from typing import Dict

from autogpt.core.messaging.base import (
    Listener,
    Message,
    MessageBroker,
    MessageChannel,
    MessageEmitter,
    MessageFilter,
    MessageMetadata,
    Role,
    Sender,
)


class SimpleMessageEmitter(MessageEmitter):
    def __init__(
        self,
        message_channel: "SimpleMessageChannel",
        sender_name: str,
        sender_role: Role,
        **additional_metadata,
    ):
        self._sender = Sender(name=sender_name, role=sender_role)
        self._additional_metadata = additional_metadata
        self._message_channel = message_channel

    async def send_message(self, content: Dict, **extra_metadata) -> bool:
        additional_metadata = {
            **self._additional_metadata,
            **extra_metadata,
        }

        metadata = MessageMetadata(
            sender=self._sender,
            timestamp=datetime.now(),
            additional_metadata=additional_metadata,
        )
        message = Message(
            content=content,
            metadata=metadata,
        )
        return await self._message_channel.send_message(message)

    def __repr__(self) -> str:
        return f"SimpleMessageEmitter(sender={self._sender}, channel={self._message_channel})"


class SimpleMessageChannel(MessageChannel):
    def __init__(
        self,
        channel_name: str,
        message_broker: "MessageBroker",
    ):
        self._channel_name = channel_name
        self._message_broker = message_broker
        self._listeners = []

    def add_listener(
        self,
        listener: Listener,
        message_filter: MessageFilter,
    ) -> None:
        self._listeners.append((listener, message_filter))

    async def send_message(self, message: Message) -> bool:
        for listener, message_filter in self._listeners:
            if filtered_message := message_filter(message):
                await listener(filtered_message)
        return True

    def __repr__(self) -> str:
        listeners = [listener.__name__ for listener, _ in self._listeners]
        return f"SimpleMessageChannel(name={self._channel_name}, listeners={listeners})"


class SimpleMessageBroker(MessageBroker):
    def __init__(self):
        self._channels: Dict[str, SimpleMessageChannel] = {}

    def create_message_channel(
        self,
        channel_name: str,
    ) -> None:
        if channel_name not in self._channels:
            self._channels[channel_name] = SimpleMessageChannel(
                channel_name=channel_name,
                message_broker=self,
            )

    def get_emitter(
        self,
        channel_name: str,
        sender_name: str,
        sender_role: Role,
        **extra_metadata,
    ) -> SimpleMessageEmitter:
        channel = self._channels[channel_name]
        return SimpleMessageEmitter(
            message_channel=channel,
            sender_name=sender_name,
            sender_role=sender_role,
            **extra_metadata,
        )

    def register_listener(
        self,
        message_channel: str,
        listener: Listener,
        message_filter: MessageFilter = lambda x: True,
    ) -> None:
        self._channels[message_channel].add_listener(listener, message_filter)

    def __repr__(self) -> str:
        return f"SimpleMessageBroker(channels={self._channels})"
