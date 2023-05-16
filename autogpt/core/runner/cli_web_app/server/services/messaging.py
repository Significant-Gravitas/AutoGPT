from asyncio.queues import Queue
from collections import defaultdict

# Set up MessageBroker shim as a thin intermediary between the HTTP server and the agent.
# This is a critical abstraction to enable robust inter-agent communication later and to
# enable long-running agents. For now, it's just an indirection layer to mock something like
# kafka, redis, or rabbitmq, e.g.


class ApplicationQueue:
    queues = defaultdict(Queue)

    class Config:
        arbitrary_types_allowed = True

    async def get(self, channel_name: str):
        """Gets a message from the channel."""
        return await self.queues[channel_name].get()

    async def put(self, channel_name: str, message) -> None:
        await self.queues[channel_name].put(message)


APPLICATION_MESSAGE_QUEUE = ApplicationQueue()
