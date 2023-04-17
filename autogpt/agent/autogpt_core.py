"""The AutoGPT core."""
import autogpt.agent.base_ai_core
from autogpt.agent.base_ai_core import (
    Abilities,
    Ability,
    BaseAICore,
    Command,
    Event,
    LLMProvider,
    MemoryProvider,
)


class AutoGptCore(autogpt.agent.base_ai_core.BaseAICore):
    """The AutoGPT core."""

    def __init__(
        self,
        memory_provider: autogpt.agent.memory_provider.MemoryProvider,
        llm_provider: autogpt.agent.llm_provider.LLMProvider,
        abilities: list[autogpt.agent.ability.Ability],
        command_channel: tuple[
            autogpt.agent.channel.Channel, autogpt.agent.channel.Channel
        ],
        event_channel: tuple[
            autogpt.agent.channel.Channel, autogpt.agent.channel.Channel
        ],
        master_mind_id: str | None = None,
        master_mind_channel: tuple[
            autogpt.agent.channel.Channel, autogpt.agent.channel.Channel
        ]
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
        pass
