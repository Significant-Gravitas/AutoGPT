from collections import defaultdict
from typing import Callable

from autogpt.core.messaging.base import MessageBroker, Message


class SimpleMessageBroker(MessageBroker):

    def __init__(self, logger):
        self._logger = logger
        self._listeners = defaultdict(list)

    def register_listener(
        self,
        message_channel: str,
        listener_callback: Callable[[Message], None],
    ) -> None:
        self._listeners[message_channel].append(listener_callback)

    def send_message(self, message_channel: str, content: dict) -> None:
        for listener in self._listeners[message_channel]:
            listener(Message(message_channel, content))
