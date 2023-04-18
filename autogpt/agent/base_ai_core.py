"""
The base class for all AI cores.

The BaseAICore class is the base class for all AI cores. 
It provides the framework for building an autonomous AI agent.
"""
from __future__ import annotations

import abc
import typing

import pydantic
import trio

from autogpt.agent.LLM.base_llm_provider import BaseLLMProvider
from autogpt.agent.memory.base_memory_provider import BaseMemoryProvider
from autogpt.agent.messages import Command, Event


class Ability(pydantic.BaseModel):
    """A ability that the AI can execute."""

    name: str
    function: typing.Callable
    arguments: typing.Dict[str, str]


class Abilities(pydantic.BaseModel):
    """A list of abilities that the AI can execute."""

    abilities: typing.List[Ability]


class BaseAICore(abc.ABC):
    """The base class for all AI cores."""

    def __init__(
        self,
        memory_provider: BaseLLMProvider,
        llm_provider: BaseMemoryProvider,
        abilities: Abilities,
        command_channel: typing.Tuple[
            trio.MemorySendChannel, trio.MemoryReceiveChannel
        ],
        event_channel: typing.Tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel],
        # The master mind is the AI that controls this AI.
        # If this AI is not controlled by another AI, then this value is None.
        master_mind_id: str = None,
        master_mind_channel: typing.Tuple[
            trio.MemorySendChannel, trio.MemoryReceiveChannel
        ] = (None, None),
    ):
        """
        Initialize a class instance.

        Args:
            memory_provider (MemoryProvider): The memory provider.
            llm_provider (LLMProvider): The LLM provider.
            abilities (Abilities): The abilities that the AI can execute.
            command_channel (typing.Tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel], optional): The command channel.
            event_channel (typing.Tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel], optional): The event channel.
            master_mind_id (str, optional): The master mind's id. Defaults to None.
            master_mind_channel (typing.Tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel], optional): The master mind's channel. Defaults to (None, None).

        """
        self.memory_provider = memory_provider
        self.llm_provider = llm_provider
        self.abilities = abilities
        self.command_send_channel, self.command_receive_channel = command_channel
        self.event_send_channel, self.event_receive_channel = event_channel

        self.is_sub_mind = master_mind_id is not None
        self.master_mind_id = master_mind_id
        self.master_send_channel, self.master_receive_channel = master_mind_channel

    async def send_event(self, event: Event) -> None:
        """Send an event to the event channel.

        Args:
            event (Event): The event to be sent.
        """
        await self.event_send_channel.send(event)

    async def receive_command(self) -> None:
        """
        Receive a command from the command channel and sends it for processing
        """
        command = await self.command_receive_channel.receive()
        self.process_commands(command)

    async def receive_command_from_master(self) -> None:
        """
        Receive a command from the master mind's command channel and sends it for processing
        """
        command = await self.master_receive_channel.receive()
        self.process_commands(command)

    async def send_report_to_master(self, command: Command) -> None:
        """
        Send a report to the master mind's event channel.
        """
        await self.master_send_channel.send(command)

    @abc.abstractmethod
    async def process_commands(self, command: Command) -> None:
        """
        Process a command.

        Args:
            command (Command): The command to be processed.
        """
        pass

    @abc.abstractmethod
    async def run(self) -> None:
        """The main loop of the AI core."""
        pass

    async def spawn_sub_mind(self, ai_core: BaseAICore) -> None:
        """
        Spawn a sub mind.

        Args:
            ai_core (BaseAICore): The AI core to be spawned.
        """
        async with trio.open_nursery() as nursery:
            nursery.start_soon(ai_core.run)
