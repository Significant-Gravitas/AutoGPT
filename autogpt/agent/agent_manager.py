"""Agent manager for managing GPT agents"""
from __future__ import annotations

import typing
from typing import Union

import pydantic
import trio

from autogpt.agent.autogpt_core import Abilities, Ability, AutoGptCore
from autogpt.agent.LLM.openai_provider import OpenAIProvider
from autogpt.agent.memory.local_memory_provider import LocalMemoryProvider
from autogpt.agent.messages import Command, Event
from autogpt.config.config import Singleton
from autogpt.llm_utils import create_chat_completion


class AgentManager(metaclass=Singleton):
    """Agent manager for managing GPT agents"""

    def __init__(self):
        self.next_key = 0
        self.agents = {}  # key, (task, full_message_history, model)

    # Create new GPT agent
    # TODO: Centralise use of create_chat_completion() to globally enforce token limit

    def create_agent(self, task: str, prompt: str, model: str) -> tuple[int, str]:
        """Create a new agent and return its key

        Args:
            task: The task to perform
            prompt: The prompt to use
            model: The model to use

        Returns:
            The key of the new agent
        """
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Start GPT instance
        agent_reply = create_chat_completion(
            model=model,
            messages=messages,
        )

        # Update full message history
        messages.append({"role": "assistant", "content": agent_reply})

        key = self.next_key
        # This is done instead of len(agents) to make keys unique even if agents
        # are deleted
        self.next_key += 1

        self.agents[key] = (task, messages, model)

        return key, agent_reply

    def message_agent(self, key: str | int, message: str) -> str:
        """Send a message to an agent and return its response

        Args:
            key: The key of the agent to message
            message: The message to send to the agent

        Returns:
            The agent's response
        """
        task, messages, model = self.agents[int(key)]

        # Add user message to message history before sending to agent
        messages.append({"role": "user", "content": message})

        # Start GPT instance
        agent_reply = create_chat_completion(
            model=model,
            messages=messages,
        )

        # Update full message history
        messages.append({"role": "assistant", "content": agent_reply})

        return agent_reply

    def list_agents(self) -> list[tuple[str | int, str]]:
        """Return a list of all agents

        Returns:
            A list of tuples of the form (key, task)
        """

        # Return a list of agent keys and their tasks
        return [(key, task) for key, (task, _, _) in self.agents.items()]

    def delete_agent(self, key: Union[str, int]) -> bool:
        """Delete an agent from the agent manager

        Args:
            key: The key of the agent to delete

        Returns:
            True if successful, False otherwise
        """

        try:
            del self.agents[int(key)]
            return True
        except KeyError:
            return False


def main(
    command_send: trio.MemorySendChannel,
    command_receive: trio.MemoryReceiveChannel,
    event_send: trio.MemorySendChannel,
    event_receive: trio.MemoryReceiveChannel,
) -> AutoGptCore:
    """Main async function"""
    # Create memory and LLM providers.
    memory_provider = LocalMemoryProvider()
    llm_provider = OpenAIProvider("", "")

    # Define AI abilities.
    abilities = Abilities(abilities=[])

    # Instantiate the AI core.
    ai_core = AutoGptCore(
        memory_provider=memory_provider,
        llm_provider=llm_provider,
        abilities=abilities,
        command_channel=(command_send, command_receive),
        event_channel=(event_send, event_receive),
    )

    # Run the AI core.
    return ai_core


async def tester(
    command_send: trio.MemorySendChannel,
    command_receive: trio.MemoryReceiveChannel,
    event_send: trio.MemorySendChannel,
    event_receive: trio.MemoryReceiveChannel,
) -> None:
    """Test function"""
    while True:
        await trio.sleep(1)
        await command_send.send(
            Command(ai_core_id="user", command="test", arguments={})
        )
        await trio.sleep(1)
        event = await event_receive.receive()
        print(f"Event: {event}")


async def async_agent_manager() -> None:
    """Return the agent manager instance"""
    from autogpt.api.server import app

    # Create command and event channels.
    command_send, command_receive = trio.open_memory_channel(0)
    event_send, event_receive = trio.open_memory_channel(0)

    ai_core = main(command_send, command_receive, event_send, event_receive)
    # Run FastAPI server and AI agent concurrently.
    async with trio.open_nursery() as nursery:
        nursery.start_soon(ai_core.run)
        nursery.start_soon(
            tester, command_send, command_receive, event_send, event_receive
        )
