from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Generic, Iterator, TypeVar

from pydantic import Field
from pydantic.generics import GenericModel

from forge.content_processing.text import summarize_text
from forge.llm.prompting.utils import format_numbered_list, indent
from forge.models.action import ActionProposal, ActionResult
from forge.models.utils import ModelWithSummary

if TYPE_CHECKING:
    from forge.config.config import Config
    from forge.llm.providers import ChatModelProvider

AP = TypeVar("AP", bound=ActionProposal)


class Episode(GenericModel, Generic[AP]):
    action: AP
    result: ActionResult | None
    summary: str | None = None

    def format(self):
        step = f"Executed `{self.action.use_tool}`\n"
        reasoning = (
            _r.summary()
            if isinstance(_r := self.action.thoughts, ModelWithSummary)
            else _r
        )
        step += f'- **Reasoning:** "{reasoning}"\n'
        step += (
            "- **Status:** "
            f"`{self.result.status if self.result else 'did_not_finish'}`\n"
        )
        if self.result:
            if self.result.status == "success":
                result = str(self.result)
                result = "\n" + indent(result) if "\n" in result else result
                step += f"- **Output:** {result}"
            elif self.result.status == "error":
                step += f"- **Reason:** {self.result.reason}\n"
                if self.result.error:
                    step += f"- **Error:** {self.result.error}\n"
            elif self.result.status == "interrupted_by_human":
                step += f"- **Feedback:** {self.result.feedback}\n"
        return step

    def __str__(self) -> str:
        executed_action = f"Executed `{self.action.use_tool}`"
        action_result = f": {self.result}" if self.result else "."
        return executed_action + action_result


class EpisodicActionHistory(GenericModel, Generic[AP]):
    """Utility container for an action history"""

    episodes: list[Episode[AP]] = Field(default_factory=list)
    cursor: int = 0
    _lock = asyncio.Lock()

    @property
    def current_episode(self) -> Episode[AP] | None:
        if self.cursor == len(self):
            return None
        return self[self.cursor]

    def __getitem__(self, key: int) -> Episode[AP]:
        return self.episodes[key]

    def __iter__(self) -> Iterator[Episode[AP]]:
        return iter(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __bool__(self) -> bool:
        return len(self.episodes) > 0

    def register_action(self, action: AP) -> None:
        if not self.current_episode:
            self.episodes.append(Episode(action=action, result=None))
            assert self.current_episode
        elif self.current_episode.action:
            raise ValueError("Action for current cycle already set")

    def register_result(self, result: ActionResult) -> None:
        if not self.current_episode:
            raise RuntimeError("Cannot register result for cycle without action")
        elif self.current_episode.result:
            raise ValueError("Result for current cycle already set")

        self.current_episode.result = result
        self.cursor = len(self.episodes)

    def rewind(self, number_of_episodes: int = 0) -> None:
        """Resets the history to an earlier state.

        Params:
            number_of_cycles (int): The number of cycles to rewind. Default is 0.
                When set to 0, it will only reset the current cycle.
        """
        # Remove partial record of current cycle
        if self.current_episode:
            if self.current_episode.action and not self.current_episode.result:
                self.episodes.pop(self.cursor)

        # Rewind the specified number of cycles
        if number_of_episodes > 0:
            self.episodes = self.episodes[:-number_of_episodes]
            self.cursor = len(self.episodes)

    async def handle_compression(
        self, llm_provider: ChatModelProvider, app_config: Config
    ) -> None:
        """Compresses each episode in the action history using an LLM.

        This method iterates over all episodes in the action history without a summary,
        and generates a summary for them using an LLM.
        """
        compress_instruction = (
            "The text represents an action, the reason for its execution, "
            "and its result. "
            "Condense the action taken and its result into one line. "
            "Preserve any specific factual information gathered by the action."
        )
        async with self._lock:
            # Gather all episodes without a summary
            episodes_to_summarize = [ep for ep in self.episodes if ep.summary is None]

            # Parallelize summarization calls
            summarize_coroutines = [
                summarize_text(
                    episode.format(),
                    instruction=compress_instruction,
                    llm_provider=llm_provider,
                    config=app_config,
                )
                for episode in episodes_to_summarize
            ]
            summaries = await asyncio.gather(*summarize_coroutines)

            # Assign summaries to episodes
            for episode, (summary, _) in zip(episodes_to_summarize, summaries):
                episode.summary = summary

    def fmt_list(self) -> str:
        return format_numbered_list(self.episodes)

    def fmt_paragraph(self) -> str:
        steps: list[str] = []

        for i, episode in enumerate(self.episodes, 1):
            step = f"### Step {i}: {episode.format()}\n"

            steps.append(step)

        return "\n\n".join(steps)
