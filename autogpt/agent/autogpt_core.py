"""The AutoGPT core."""
import trio

from autogpt.agent.base_ai_core import Abilities, Ability, BaseAICore
from autogpt.agent.LLM.openai_provider import OpenAIProvider
from autogpt.agent.memory.local_memory_provider import LocalMemoryProvider


class AutoGptCore(BaseAICore):
    """The AutoGPT core."""

    def __init__(
        self,
        memory_provider: LocalMemoryProvider,
        llm_provider: OpenAIProvider,
        abilities: list[Ability],
        command_channel: tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel],
        event_channel: tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel],
        master_mind_id: str | None = None,
        master_mind_channel: tuple[trio.MemorySendChannel, trio.MemoryReceiveChannel]
        | None = None,
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
            command = await self.command_receive_channel.receive()
            # wait 1 second
            await trio.sleep(1)
            print(command)
