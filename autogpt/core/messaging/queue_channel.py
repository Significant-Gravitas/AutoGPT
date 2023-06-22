"""Basic in memory implementation using Asyncio queues."""
import asyncio.queues

from autogpt.core.schema import BaseEvent, BaseMessageChannel


class QueueChannel(BaseMessageChannel):
    """A queue channel that messages can be sent and received on."""

    queue = asyncio.queues.Queue()

    class Config:
        """
        Config class is a configuration class for Pydantic models.
        """

        arbitrary_types_allowed = True

    async def get(self) -> BaseEvent:
        """Gets a message from the channel."""
        msg = await self.queue.get()
        self.received_message_count += 1
        self.received_bytes_count += msg.__sizeof__()
        return msg

    async def send(self, message: BaseEvent) -> None:
        """Sends a message to the channel."""
        self.sent_message_count += 1
        self.sent_bytes_count += message.__sizeof__()
        await self.queue.put(message)
