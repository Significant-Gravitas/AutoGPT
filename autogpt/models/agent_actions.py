from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional

from autogpt.prompts.utils import format_numbered_list


@dataclass
class Action:
    name: str
    args: dict[str, Any]
    reasoning: str

    def format_call(self) -> str:
        return f"{self.name}({', '.join([f'{a}={repr(v)}' for a, v in self.args.items()])})"


@dataclass
class ActionSuccessResult:
    results: Any
    status: Literal["success"] = "success"

    def __str__(self) -> str:
        results = (
            f'"""{self.results}"""'
            if type(self.results) == str and any(s in self.results for s in ("\n", '"'))
            else f'"{self.results}"'
        )
        return f"Action succeeded, and returned: {results}"


@dataclass
class ActionErrorResult:
    reason: str
    error: Optional[Exception] = None
    status: Literal["error"] = "error"

    def __str__(self) -> str:
        return f"Action failed: '{self.reason}'"


@dataclass
class ActionInterruptedByHuman:
    feedback: str
    status: Literal["interrupted_by_human"] = "interrupted_by_human"

    def __str__(self) -> str:
        return f'The user interrupted the action with the following feedback: "{self.feedback}"'


ActionResult = ActionSuccessResult | ActionErrorResult | ActionInterruptedByHuman


class ActionHistory:
    """Utility container for an action history"""

    @dataclass
    class CycleRecord:
        action: Action
        result: ActionResult | None

        def __str__(self):
            executed_action = f"You executed `{self.action.format_call()}`."
            action_result = f" Result: {self.result}" if self.result else ""
            return executed_action + action_result

    cursor: int
    cycles: list[CycleRecord]

    def __init__(self, cycles: list[CycleRecord] = []):
        self.cycles = cycles
        self.cursor = len(self.cycles)

    @property
    def current_record(self) -> CycleRecord | None:
        if self.cursor == len(self):
            return None
        return self[self.cursor]

    def __getitem__(self, key: int) -> CycleRecord:
        return self.cycles[key]

    def __iter__(self) -> Iterator[CycleRecord]:
        return iter(self.cycles)

    def __len__(self) -> int:
        return len(self.cycles)

    def __bool__(self) -> bool:
        return len(self.cycles) > 0

    def register_action(self, action: Action) -> None:
        if not self.current_record:
            self.cycles.append(self.CycleRecord(action, None))
            assert self.current_record
        elif self.current_record.action:
            raise ValueError("Action for current cycle already set")

    def register_result(self, result: ActionResult) -> None:
        if not self.current_record:
            raise RuntimeError("Cannot register result for cycle without action")
        elif self.current_record.result:
            raise ValueError("Result for current cycle already set")

        self.current_record.result = result
        self.cursor = len(self.cycles)

    def generate_list(self) -> str:
        return format_numbered_list(self.cycles)
