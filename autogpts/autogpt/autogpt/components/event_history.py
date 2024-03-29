from typing import Callable, Iterator, Optional

from autogpts.autogpt.autogpt.agents.components import Component
from autogpts.autogpt.autogpt.agents.protocols import MessageProvider
from autogpts.autogpt.autogpt.core.resource.model_providers.schema import ChatMessage
from autogpts.autogpt.autogpt.models.action_history import (
    Episode,
    EpisodicActionHistory,
)
from autogpt.prompts.utils import indent


class EventHistoryComponent(Component, MessageProvider):
    def __init__(
        self,
        event_history: EpisodicActionHistory,
        max_tokens: int,
        count_tokens: Callable[[str], int],
    ) -> None:
        self.event_history = event_history
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens

    def get_messages(self) -> Iterator[ChatMessage]:
        yield ChatMessage.system(
            self._compile_progress(
                self.event_history.episodes,
                self.max_tokens,
                self.count_tokens,
            )
        )

    def _compile_progress(
        self,
        episode_history: list[Episode],
        max_tokens: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None,
    ) -> str:
        if max_tokens and not count_tokens:
            raise ValueError("count_tokens is required if max_tokens is set")

        steps: list[str] = []
        tokens: int = 0
        n_episodes = len(episode_history)

        for i, episode in enumerate(reversed(episode_history)):
            # Use full format for the latest 4 steps, summary or format for older steps
            if i < 4 or episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            if max_tokens and count_tokens:
                step_tokens = count_tokens(step)
                if tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens

            steps.insert(0, step)

        return "\n\n".join(steps)
