"""What a tool call reads as when its turn is stopped mid-call.

A stopped turn closes every unresolved tool call with a synthetic result so
the transcript stays valid. With nothing stashed that result was ``""``,
which a reader cannot tell apart from a call still in flight — a delegation
cut off by Stop reloaded as "working" forever. A tool that knows what being
stopped means for it (a delegation: "the teammate's task was stopped too")
records that here while it waits, and the flush uses it instead.

Keyed like the output stash (tool name + the model's own arguments), so each
call finds its own record. The registry dict is created per turn by
:func:`reset_cancelled_outputs` before any tool task starts, so every tool
task shares it by reference, like the output stash.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_outputs: ContextVar[dict[str, str] | None] = ContextVar(
    "cancelled_tool_outputs", default=None
)
_call_key: ContextVar[str | None] = ContextVar(
    "cancelled_output_call_key", default=None
)


def reset_cancelled_outputs() -> None:
    """Start a turn (or a retry attempt) with no recorded results."""
    _outputs.set({})


@contextmanager
def tool_call(key: str) -> Iterator[None]:
    """Scope a tool call so what it records lands under *key*.

    A call that returns normally has its real output, so its record is
    dropped on the way out; only a call cut off mid-flight keeps one.
    """
    token = _call_key.set(key)
    try:
        yield
        _discard(key)
    finally:
        _call_key.reset(token)


def record_cancelled_output(payload: str) -> None:
    """Say what the running call's result is if its turn stops now."""
    outputs, key = _outputs.get(), _call_key.get()
    if outputs is not None and key is not None:
        outputs[key] = payload


def pop_cancelled_output(key: str) -> str | None:
    outputs = _outputs.get()
    return outputs.pop(key, None) if outputs is not None else None


def _discard(key: str) -> None:
    outputs = _outputs.get()
    if outputs is not None:
        outputs.pop(key, None)
