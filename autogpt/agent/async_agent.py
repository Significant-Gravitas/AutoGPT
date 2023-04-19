"""An asynchronous agent that can perform tasks and send and receive messages."""
import asyncio
import random
from typing import List, Optional, Tuple

import trio
from pydantic import BaseModel


class Message(BaseModel):
    """A simple message class for sending and receiving messages."""

    recipient_id: str
    content: str


class EventQueue:
    """A simple event queue for sending and receiving messages."""

    def __init__(self):
        self.queue: List[Tuple[str, str]] = []
        self.agents: set[str] = set()

    def add_agent(self, agent_id: str) -> None:
        """Add an agent to the event queue.

        Args:
            agent_id (str): The agent's identifier.
        """
        self.agents.add(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the event queue.

        Args:
            agent_id (str): The agent's identifier.
        """
        self.agents.remove(agent_id)

    def send_message(self, message: Message) -> None:
        """Add a message to the queue.

        Args:
            message (Message): The message to be added to the queue.
        """
        self.queue.append((message.recipient_id, message.content))

    def get_message(self, recipient_id: str) -> Optional[str]:
        """Retrieve a message for the specified recipient from the queue.

        Args:
            recipient_id (str): The recipient's identifier.

        Returns:
            Optional[str]: The message content if found, otherwise None.
        """
        for i, (recv_id, msg) in enumerate(self.queue):
            if recv_id == recipient_id:
                del self.queue[i]
                return msg
        return None


class AsyncAgent:
    """An asynchronous agent that can perform tasks and send and receive messages."""

    def __init__(self, name: str, event_queue: EventQueue):
        self.name = name
        self.event_queue = event_queue
        self.event_queue.add_agent(self.name)
        self.sub_mind_count = 0

    async def roll_dice(self) -> int:
        """Roll a dice and return a random number between 1 and 6.

        Returns:
            int: The result of the dice roll.
        """
        return random.randint(1, 4)

    async def check_messages(self) -> None:
        """Check the event queue for messages addressed to the agent and process them."""
        if message := self.event_queue.get_message(self.name):
            print(f"{self.name}: MESSAGE received: {message}")

    async def perform_task(self, task_name: str, duration: int) -> None:
        """Perform a task for a specified duration and take actions based on a dice roll.

        Args:
            task_name (str): The name of the task.
            duration (int): The duration of the task in seconds.
        """
        print(f"{self.name} starts {task_name}")

        for _ in range(duration):
            await self.check_messages()
            await asyncio.sleep(1)

        dice_result = await self.roll_dice()
        print(f"{self.name} rolled a {dice_result}")

        if dice_result == 1:
            print(f"{self.name} finishes {task_name}")
            self.event_queue.remove_agent(self.name)
        elif dice_result == 2:
            print(f"{self.name} starts another task")
            await self.perform_task(f"Another task for {task_name}", duration)
        elif dice_result == 3:
            self.sub_mind_count += 1
            new_agent_name = f"{self.name}-Sub-Mind-{self.sub_mind_count}"
            new_agent = AsyncAgent(new_agent_name, self.event_queue)
            print(f"{self.name} spawns {new_agent_name} and starts another task")
            await asyncio.gather(
                self.perform_task(f"Another task for {task_name}", duration),
                new_agent.perform_task(f"Task for {new_agent_name}", duration),
            )
        elif dice_result == 4 and self.event_queue.agents:
            random_agent = random.choice(list(self.event_queue.agents))
            self.send_message(random_agent, f"{self.name} is done with {task_name}")
            await self.perform_task(f"Another task for {task_name}", duration)

    def send_message(self, recipient_id: str, content: str) -> None:
        """Send a message to another agent.

        Args:
            recipient_id (str): The recipient's identifier.
            content (str): The content
        """
        message = Message(recipient_id=recipient_id, content=content)
        self.event_queue.send_message(message)


async def main() -> None:
    """Run the main function."""
    event_queue = EventQueue()
    agent_a = AsyncAgent("Auto-GPT", event_queue)

    # Run agent_a using trio
    await trio.to_thread.run_sync(
        asyncio.run,
        agent_a.perform_task("Thinking...", 1),
    )


if __name__ == "__main__":
    trio.run(main)
