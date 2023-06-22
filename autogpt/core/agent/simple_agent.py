import asyncio
import logging
import random
import time

from autogpt.core.messaging.messages import (
    CommandMessage,
    InfoMessage,
    QueryMessage,
    ResponseMessage,
    ShutdownMessage,
    StatusMessage,
)
from autogpt.core.schema import BaseAgent, BaseLLMProvider, BaseMessage

logger = logging.getLogger(__name__)


class SampleAgent(BaseAgent):
    """An async agent that can send a receive messages using a BaseMessage and process all subtypes of BaseMessage"""

    name: str
    llm_provider: BaseLLMProvider
    running: bool = True
    stopping: bool = True  # used for graceful shutdown

    async def testing_random_action(self) -> None:
        """A random action for testing purposes"""
        to_uid = "agent1"
        if self.uid == "agent1":
            to_uid = "agent2"
        roll = random.randint(1, 3)
        timestamp = int(time.time())
        if roll == 1:
            await self.message_broker.send_to_channel(
                "channel1",
                QueryMessage(
                    query="What is the meaning of life?",
                    from_uid=self.uid,
                    to_uid=to_uid,
                    timestamp=timestamp,
                ),
            )
        elif roll == 2:
            ans = await self.llm_provider.chat(
                [{"role": "user", "content": "Hello, how are you?"}]
            )
            print(ans)

    async def run(self) -> None:
        """Runs the agent"""
        try:
            while self.running:
                print(f"Agent: {self.uid} is running...")

                if asyncio.current_task().cancelled():
                    logger.warning("Agent cancelled Abruptly with no graceful shutdown")
                    break

                # Perform a random action
                await self.testing_random_action()
                for channel in self.message_broker.channels:
                    message = await channel.get()
                    await self.process_message(message)
                    if self.stopping:
                        await self.shutdown()
                        return

                # sleep 1 second
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.warning(f"Agent: {self.name}:{self.uid} stopped suddenly")

    async def process_message(self, message: BaseMessage) -> None:
        """Process an incoming message"""
        if isinstance(message, QueryMessage):
            await self.process_query(message)
        elif isinstance(message, ShutdownMessage):
            await self.stop()
        elif isinstance(message, CommandMessage):
            await self.process_command(message)
        elif isinstance(message, ResponseMessage):
            await self.process_response(message)
        elif isinstance(message, StatusMessage):
            await self.process_status(message)
        elif isinstance(message, InfoMessage):
            await self.process_info(message)
        else:
            print(f"Unknown message type: {type(message)}")

    async def process_query(self, message: QueryMessage) -> None:
        """Process a QueryMessage"""
        print(f"Agent: {self.uid} received query: {message.query}")

    async def process_command(self, message: CommandMessage) -> None:
        """Process a CommandMessage"""
        print(f"Agent: {self.uid} received command: {message.command}")

    async def process_response(self, message: ResponseMessage) -> None:
        """Process a ResponseMessage"""
        print(f"Agent: {self.uid} received response: {message.response}")

    async def process_status(self, message: StatusMessage) -> None:
        """Process a StatusMessage"""
        print(f"Agent: {self.uid} received status: {message.status}")

    async def process_info(self, message: InfoMessage) -> None:
        """Process an InfoMessage"""
        print(f"Agent: {self.uid} received info: {message.info}")

    async def stop(self) -> None:
        """stop the agent"""
        self.running = False

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        logger.info("Gracefully shutting down the agent")
