from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Generic

from pydantic import BaseModel, Field

from forge.content_processing.text import summarize_text
from forge.llm.prompting.utils import format_numbered_list, indent
from forge.llm.providers.multi import ModelName
from forge.models.action import ActionResult, AnyProposal
from forge.models.utils import ModelWithSummary

if TYPE_CHECKING:
    from forge.llm.providers import MultiProvider

logger = logging.getLogger(__name__)


class Episode(BaseModel, Generic[AnyProposal]):
    action: AnyProposal
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


class EpisodicActionHistory(BaseModel, Generic[AnyProposal]):
    """Utility container for an action history"""

    episodes: list[Episode[AnyProposal]] = Field(default_factory=list)
    cursor: int = 0
    pending_user_feedback: list[str] = Field(default_factory=list)
    """User feedback provided along with approval, for inclusion in next prompt"""
    _lock = asyncio.Lock()

    @property
    def current_episode(self) -> Episode[AnyProposal] | None:
        if self.cursor == len(self):
            return None
        return self[self.cursor]

    def __getitem__(self, key: int) -> Episode[AnyProposal]:
        return self.episodes[key]

    def __len__(self) -> int:
        return len(self.episodes)

    def __bool__(self) -> bool:
        return len(self.episodes) > 0

    def register_action(self, action: AnyProposal) -> None:
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

    def append_user_feedback(self, feedback: str) -> None:
        """Append user feedback to be included in the next prompt.

        This is used when a user approves a command but also provides feedback.
        The feedback will be sent to the agent in the next iteration.

        Args:
            feedback: The user's feedback text.
        """
        self.pending_user_feedback.append(feedback)

    def pop_pending_feedback(self) -> list[str]:
        """Get and clear all pending user feedback.

        Returns:
            List of feedback strings that were pending.
        """
        feedback = self.pending_user_feedback.copy()
        self.pending_user_feedback.clear()
        return feedback

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
        self,
        llm_provider: MultiProvider,
        model_name: ModelName,
        spacy_model: str,
        full_message_count: int = 4,
    ) -> None:
        """Compresses older episodes in the action history using an LLM.

        Only episodes older than `full_message_count` are compressed, since recent
        episodes are shown in full in the prompt anyway.

        Args:
            llm_provider: LLM provider for summarization
            model_name: Model to use for summarization
            spacy_model: Spacy model for text chunking
            full_message_count: Number of recent episodes to skip (shown in full)
        """
        compress_instruction = (
            "The text represents an action, the reason for its execution, "
            "and its result. "
            "Condense the action taken and its result into one line. "
            "Preserve any specific factual information gathered by the action."
        )
        async with self._lock:
            # Only compress episodes older than full_message_count
            # Recent episodes are shown in full, so no need to summarize them
            n_episodes = len(self.episodes)
            if n_episodes <= full_message_count:
                return  # All episodes are recent, no compression needed

            # Get older episodes that need compression
            older_episodes = self.episodes[: n_episodes - full_message_count]
            episodes_to_summarize = [ep for ep in older_episodes if ep.summary is None]

            if not episodes_to_summarize:
                return  # No episodes need compression

            logger.debug(
                f"Compressing {len(episodes_to_summarize)} action history episodes "
                f"(total: {n_episodes}, full_message_count: {full_message_count})"
            )

            # Parallelize summarization calls
            summarize_coroutines = [
                summarize_text(
                    episode.format(),
                    instruction=compress_instruction,
                    llm_provider=llm_provider,
                    model_name=model_name,
                    spacy_model=spacy_model,
                )
                for episode in episodes_to_summarize
            ]
            summaries = await asyncio.gather(*summarize_coroutines)

            # Assign summaries to episodes
            for episode, (summary, _) in zip(episodes_to_summarize, summaries):
                episode.summary = summary

            logger.debug(
                f"Compression complete for {len(episodes_to_summarize)} episodes"
            )

    def fmt_list(self) -> str:
        return format_numbered_list(self.episodes)

    def fmt_paragraph(self) -> str:
        steps: list[str] = []

        for i, episode in enumerate(self.episodes, 1):
            step = f"### Step {i}: {episode.format()}\n"

            steps.append(step)

        return "\n\n".join(steps)
