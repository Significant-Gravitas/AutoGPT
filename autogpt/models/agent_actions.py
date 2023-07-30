from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


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
        return f"Action succeeded and returned: `{self.results}`"


@dataclass
class ActionErrorResult:
    reason: str
    error: Optional[Exception] = None
    status: Literal["error"] = "error"

    def __str__(self) -> str:
        return f"Action failed: `{self.reason}`"


@dataclass
class ActionInterruptedByHuman:
    feedback: str
    status: Literal["interrupted_by_human"] = "interrupted_by_human"

    def __str__(self) -> str:
        return f'The user interrupted the action with the following feedback: "{self.feedback}"'


ActionResult = ActionSuccessResult | ActionErrorResult | ActionInterruptedByHuman
