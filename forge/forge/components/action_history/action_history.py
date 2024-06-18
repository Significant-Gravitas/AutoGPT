from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator, Optional

from forge.agent.protocols import AfterExecute, AfterParse, MessageProvider
from forge.llm.prompting.utils import indent
from forge.llm.providers import ChatMessage, MultiProvider
from forge.llm.providers.schema import ToolResultMessage

if TYPE_CHECKING:
    from forge.config.config import Config

from .model import ActionResult, AnyProposal, Episode, EpisodicActionHistory


class ActionHistoryComponent(MessageProvider, AfterParse[AnyProposal], AfterExecute):
    """Keeps track of the event history and provides a summary of the steps."""

    def __init__(
        self,
        event_history: EpisodicActionHistory[AnyProposal],
        max_tokens: int,
        count_tokens: Callable[[str], int],
        legacy_config: Config,
        llm_provider: MultiProvider,
    ) -> None:
        self.event_history = event_history
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens
        self.legacy_config = legacy_config
        self.llm_provider = llm_provider

    def get_messages(self) -> Iterator[ChatMessage]:
        messages: list[ChatMessage] = []
        step_summaries: list[str] = []
        tokens: int = 0
        n_episodes = len(self.event_history.episodes)

        # Include a summary for all except the latest 4 steps
        for i, episode in enumerate(reversed(self.event_history.episodes)):
            # Use full format for the latest 4 steps, summary or format for older steps
            if i < 4:
                messages.insert(0, episode.action.raw_message)
                tokens += self.count_tokens(str(messages[0]))  # HACK
                if episode.result:
                    result_message = self._make_result_message(episode, episode.result)
                    messages.insert(1, result_message)
                    tokens += self.count_tokens(str(result_message))  # HACK
                continue
            elif episode.summary is None:
                step_content = indent(episode.format(), 2).strip()
            else:
                step_content = episode.summary

            step = f"* Step {n_episodes - i}: {step_content}"

            if self.max_tokens and self.count_tokens:
                step_tokens = self.count_tokens(step)
                if tokens + step_tokens > self.max_tokens:
                    break
                tokens += step_tokens

            step_summaries.insert(0, step)

        if step_summaries:
            step_summaries_fmt = "\n\n".join(step_summaries)
            yield ChatMessage.system(
                f"## Progress on your Task so far\n"
                "Here is a summary of the steps that you have executed so far, "
                "use this as your consideration for determining the next action!\n"
                f"{step_summaries_fmt}"
            )

        yield from messages

    def after_parse(self, result: AnyProposal) -> None:
        self.event_history.register_action(result)

    async def after_execute(self, result: ActionResult) -> None:
        self.event_history.register_result(result)
        await self.event_history.handle_compression(
            self.llm_provider, self.legacy_config
        )

    @staticmethod
    def _make_result_message(episode: Episode, result: ActionResult) -> ChatMessage:
        if result.status == "success":
            return (
                ToolResultMessage(
                    content=str(result.outputs),
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if episode.action.raw_message.tool_calls
                else ChatMessage.user(
                    f"{episode.action.use_tool.name} returned: "
                    + (
                        f"```\n{result.outputs}\n```"
                        if "\n" in str(result.outputs)
                        else f"`{result.outputs}`"
                    )
                )
            )
        elif result.status == "error":
            return (
                ToolResultMessage(
                    content=f"{result.reason}\n\n{result.error or ''}".strip(),
                    is_error=True,
                    tool_call_id=episode.action.raw_message.tool_calls[0].id,
                )
                if episode.action.raw_message.tool_calls
                else ChatMessage.user(
                    f"{episode.action.use_tool.name} raised an error: ```\n"
                    f"{result.reason}\n"
                    "```"
                )
            )
        else:
            return ChatMessage.user(result.feedback)

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
