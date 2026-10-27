"""What a turn wrote for a teammate to read: the text of each delegation or
handoff call, read through the ``run_capability`` wrapper too."""

import json
from typing import Any

from backend.copilot.eval.style.generation import (
    CAPABILITY_WRAPPERS,
    DELEGATION_TOOLS,
    HANDOFF_TOOL,
    Turn,
    called_name,
)

_TASK_FIELDS = ("system_context", "context", "prompt")


def delegated_tasks(turn: Turn) -> list[str]:
    """One entry per delegation or handoff call the turn made."""
    return [task for name, args in turn.calls if (task := delegated_task(name, args))]


def delegated_task(name: str, arguments: str) -> str | None:
    """The text a delegation or handoff call hands its teammate, or ``None``
    for any other call."""
    if called_name(name, arguments) not in (*DELEGATION_TOOLS, HANDOFF_TOOL):
        return None
    args = _object(arguments)
    if name in CAPABILITY_WRAPPERS:
        args = _object(args.get("input"))
    return "\n\n".join(str(args[f]) for f in _TASK_FIELDS if args.get(f)) or None


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}
