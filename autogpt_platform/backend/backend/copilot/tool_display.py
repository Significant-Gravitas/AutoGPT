"""Presentation metadata published by the current tool execution."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_display_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
    "tool_display_callback", default=None
)


@contextmanager
def tool_display_context(on_display_name: Callable[[str], None]) -> Iterator[None]:
    """Scope display updates to one tool task, including parallel executions."""
    token = _display_callback.set(on_display_name)
    try:
        yield
    finally:
        _display_callback.reset(token)


def emit_tool_display_name(display_name: str) -> None:
    """Publish a resolved name without coupling tools to the stream transport."""
    callback = _display_callback.get()
    name = display_name.strip()
    if callback is not None and name:
        callback(name)


def tool_calls_for_provider(tool_calls: list[dict]) -> list[dict]:
    """Keep UI metadata out of the provider's tool-call protocol."""
    return [
        {key: value for key, value in call.items() if key != "display_name"}
        for call in tool_calls
    ]
