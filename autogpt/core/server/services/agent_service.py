from autogpt.core.schema import BaseAgent
from autogpt.core.messaging.queue_channel import QueueChannel
import asyncio


async def create_agent(agent: BaseAgent, queue_channel: QueueChannel):

    if len(agent.message_broker.channels) == 0:
        agent.message_broker.add_channel(queue_channel)

    asyncio.create_task(agent.run())