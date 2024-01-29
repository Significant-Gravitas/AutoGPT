from __future__ import annotations

from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from autogpt.prompts.utils import format_numbered_list, indent


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

    def __str__(self) -> str:
        executed_action = f"Executed `{self.action.format_call()}`"
        action_result = f": {self.result}" if self.result else "."
        return executed_action + action_result


class EpisodicActionHistory(BaseModel):
    """Utility container for an action history"""

    episodes: list[Episode] = Field(default_factory=list)
    cursor: int = 0

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

    def fmt_list(self) -> str:
        return format_numbered_list(self.episodes)

    def fmt_paragraph(self) -> str:
        steps: list[str] = []

        for i, c in enumerate(self.episodes, 1):
            step = f"### Step {i}: Executed `{c.action.format_call()}`\n"
            step += f'- **Reasoning:** "{c.action.reasoning}"\n'
            step += (
                f"- **Status:** `{c.result.status if c.result else 'did_not_finish'}`\n"
            )
            if c.result:
                if c.result.status == "success":
                    result = str(c.result)
                    result = "\n" + indent(result) if "\n" in result else result
                    step += f"- **Output:** {result}"
                elif c.result.status == "error":
                    step += f"- **Reason:** {c.result.reason}\n"
                    if c.result.error:
                        step += f"- **Error:** {c.result.error}\n"
                elif c.result.status == "interrupted_by_human":
                    step += f"- **Feedback:** {c.result.feedback}\n"

            steps.append(step)

        return "\n\n".join(steps)
