from typing import TYPE_CHECKING, Callable, Generic, Iterator, Optional

from forge.agent.protocols import AfterExecute, AfterParse, MessageProvider
from forge.components.watchdog import WatchdogComponent
from forge.llm.prompting.utils import indent
from forge.llm.providers import ChatMessage, ChatModelProvider

if TYPE_CHECKING:
    from forge.config.config import Config

from .model import AP, ActionResult, Episode, EpisodicActionHistory


class EventHistoryComponent(MessageProvider, AfterParse, AfterExecute, Generic[AP]):
    """Keeps track of the event history and provides a summary of the steps."""

    run_after = [WatchdogComponent]

    def __init__(
        self,
        event_history: EpisodicActionHistory[AP],
        max_tokens: int,
        count_tokens: Callable[[str], int],
        legacy_config: "Config",
        llm_provider: ChatModelProvider,
    ) -> None:
        self.event_history = event_history
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens
        self.legacy_config = legacy_config
        self.llm_provider = llm_provider

    def get_messages(self) -> Iterator[ChatMessage]:
        if progress := self._compile_progress(
            self.event_history.episodes,
            self.max_tokens,
            self.count_tokens,
        ):
            yield ChatMessage.system(f"## Progress on your Task so far\n\n{progress}")

    def after_parse(self, result: AP) -> None:
        self.event_history.register_action(result)

    async def after_execute(self, result: ActionResult) -> None:
        self.event_history.register_result(result)
        await self.event_history.handle_compression(
            self.llm_provider, self.legacy_config
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
