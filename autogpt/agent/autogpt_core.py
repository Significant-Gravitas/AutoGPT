"""The AutoGPT core."""
import trio

from autogpt.agent.base_ai_core import Abilities, Ability, BaseAICore
from autogpt.agent.LLM.openai_provider import OpenAIProvider
from autogpt.agent.memory.local_memory_provider import LocalMemoryProvider
from autogpt.agent.messages import Command, Event


class AutoGptCore(BaseAICore):
    """The AutoGPT core."""

    def __init__(
        self,
        memory_provider: LocalMemoryProvider,
        llm_provider: OpenAIProvider,
        abilities: list[Ability],
        command_channel: tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel],
        event_channel: tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel],
        master_mind_id: str = None,
        master_mind_channel: tuple[
            trio.MemorySendChannel, trio.MemoryReceiveChannel
        ] = (None, None),
    ):
        super().__init__(
            memory_provider,
            llm_provider,
            abilities,
            command_channel,
            event_channel,
            master_mind_id,
            master_mind_channel,
        )

    async def run(self) -> None:
        """Run the AI core."""

        while True:
            # wait 1 second
            await self.receive_command()
            await trio.sleep(1)

    async def process_commands(self, command: Command) -> None:
        """Process a command."""
        if command.command == "test":
            await self.send_event(
                Event(
                    ai_core_id="user",
                    event="test",
                    data={"message": "received Command: test"},
                )
            )
        print(command)
