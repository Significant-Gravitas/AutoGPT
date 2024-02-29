from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from autogpt.processing.text import summarize_text
from autogpt.prompts.utils import format_numbered_list, indent

if TYPE_CHECKING:
    from autogpt.agents.base import CommandArgs, CommandName
    from autogpt.config.config import Config
    from autogpt.core.resource.model_providers import ChatModelProvider


class Action(BaseModel):
    name: str
    args: dict[str, Any]
    reasoning: str

    def format_call(self) -> str:
        return (
            f"{self.name}"
            f"({', '.join([f'{a}={repr(v)}' for a, v in self.args.items()])})"
        )


class ActionSuccessResult(BaseModel):
    outputs: Any
    status: Literal["success"] = "success"

    def __str__(self) -> str:
        outputs = str(self.outputs).replace("```", r"\```")
        multiline = "\n" in outputs
        return f"```\n{self.outputs}\n```" if multiline else str(self.outputs)


class ErrorInfo(BaseModel):
    args: tuple
    message: str
    exception_type: str
    repr: str

    @staticmethod
    def from_exception(exception: Exception) -> ErrorInfo:
        return ErrorInfo(
            args=exception.args,
            message=getattr(exception, "message", exception.args[0]),
            exception_type=exception.__class__.__name__,
            repr=repr(exception),
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.repr


class ActionErrorResult(BaseModel):
    reason: str
    error: Optional[ErrorInfo] = None
    status: Literal["error"] = "error"

    @staticmethod
    def from_exception(exception: Exception) -> ActionErrorResult:
        return ActionErrorResult(
            reason=getattr(exception, "message", exception.args[0]),
            error=ErrorInfo.from_exception(exception),
        )

    def __str__(self) -> str:
        return f"Action failed: '{self.reason}'"


class ActionInterruptedByHuman(BaseModel):
    feedback: str
    status: Literal["interrupted_by_human"] = "interrupted_by_human"

    def __str__(self) -> str:
        return (
            'The user interrupted the action with the following feedback: "%s"'
            % self.feedback
        )


ActionResult = ActionSuccessResult | ActionErrorResult | ActionInterruptedByHuman


class Episode(BaseModel):
    action: Action
    result: ActionResult | None
    summary: str | None = None

    def format(self):
        step = f"Executed `{self.action.format_call()}`\n"
        step += f'- **Reasoning:** "{self.action.reasoning}"\n'
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
        executed_action = f"Executed `{self.action.format_call()}`"
        action_result = f": {self.result}" if self.result else "."
        return executed_action + action_result


class EpisodicActionHistory(BaseModel):
    """Utility container for an action history"""

    episodes: list[Episode] = Field(default_factory=list)
    cursor: int = 0
    _lock = asyncio.Lock()

    @property
    def current_episode(self) -> Episode | None:
        if self.cursor == len(self):
            return None
        return self[self.cursor]

    def __getitem__(self, key: int) -> Episode:
        return self.episodes[key]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __bool__(self) -> bool:
        return len(self.episodes) > 0

    def register_action(self, action: Action) -> None:
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

    def matches_last_command(
        self, command_name: CommandName, arguments: CommandArgs
    ) -> bool:
        """Check if the last command matches the given name and arguments."""
        if len(self.episodes) > 0:
            last_command = self.episodes[-1].action
            return last_command.name == command_name and last_command.args == arguments
        return False

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
