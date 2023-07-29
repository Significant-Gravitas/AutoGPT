from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class Action:
    name: str
    args: dict[str, Any]
    reasoning: str

    def format_call(self) -> str:
        return f"{self.name}({', '.join([f'{a}={repr(v)}' for a, v in self.args.items()])})"


@dataclass
class ActionSuccessResult:
    success: Literal[True]
    results: Any


@dataclass
class ActionErrorResult:
    success: Literal[False]
    reason: str


ActionResult = ActionSuccessResult | ActionErrorResult


class ActionHistory:
    """Utility container for an action history"""

    @dataclass
    class CycleRecord:
        action: Action | None
        result: ActionResult | None

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

    def __iter__(self):
        return iter(self.cycles)

    def __len__(self):
        return len(self.cycles)

    def __bool__(self):
        return len(self.cycles) > 0

    def register_action(self, action: Action) -> None:
        if not self.current_record:
            self.cycles.append(self.CycleRecord(None, None))
            assert self.current_record
        elif self.current_record.action:
            raise ValueError("Action for current cycle already set")

        self.current_record.action = action

    def register_result(self, result: ActionResult) -> None:
        if not self.current_record:
            raise RuntimeError("Cannot register result for cycle without action")
        elif self.current_record.result:
            raise ValueError("Result for current cycle already set")

        self.current_record.result = result
