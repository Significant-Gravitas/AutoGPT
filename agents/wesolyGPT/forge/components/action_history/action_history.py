from __future__ import annotations

from typing import Callable, Iterator, Optional

from pydantic import BaseModel

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import AfterExecute, AfterParse, MessageProvider
from forge.llm.prompting.utils import indent
from forge.llm.providers import ChatMessage, MultiProvider
from forge.llm.providers.multi import ModelName
from forge.llm.providers.openai import OpenAIModelName

from .model import ActionResult, AnyProposal, Episode, EpisodicActionHistory


class ActionHistoryConfiguration(BaseModel):
    model_name: ModelName = OpenAIModelName.GPT3
    """Name of the llm model used to compress the history"""
    max_tokens: int = 1024
    """Maximum number of tokens to use up with generated history messages"""
    spacy_language_model: str = "en_core_web_sm"
    """Language model used for summary chunking using spacy"""


class ActionHistoryComponent(
    MessageProvider,
    AfterParse[AnyProposal],
    AfterExecute,
    ConfigurableComponent[ActionHistoryConfiguration],
):
    """Keeps track of the event history and provides a summary of the steps."""

    config_class = ActionHistoryConfiguration

    def __init__(
        self,
        event_history: EpisodicActionHistory[AnyProposal],
        count_tokens: Callable[[str], int],
        llm_provider: MultiProvider,
        config: Optional[ActionHistoryConfiguration] = None,
    ) -> None:
        ConfigurableComponent.__init__(self, config)
        self.event_history = event_history
        self.count_tokens = count_tokens
        self.llm_provider = llm_provider

    def get_messages(self) -> Iterator[ChatMessage]:
        if progress := self._compile_progress(
            self.event_history.episodes,
            self.config.max_tokens,
            self.count_tokens,
        ):
            yield ChatMessage.system(f"## Progress on your Task so far\n\n{progress}")

    def after_parse(self, result: AnyProposal) -> None:
        self.event_history.register_action(result)

    async def after_execute(self, result: ActionResult) -> None:
        self.event_history.register_result(result)
        await self.event_history.handle_compression(
            self.llm_provider, self.config.model_name, self.config.spacy_language_model
        )

    def _compile_progress(
        self,
        episode_history: list[Episode[AnyProposal]],
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
