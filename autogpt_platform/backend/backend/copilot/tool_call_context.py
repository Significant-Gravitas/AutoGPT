from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_call_id: ContextVar[str | None] = ContextVar("copilot_tool_call_id", default=None)


def current_tool_call_id() -> str | None:
    return _call_id.get()


@contextmanager
def tool_call_context(call_id: str) -> Iterator[None]:
    token = _call_id.set(call_id)
    try:
        yield
    finally:
        _call_id.reset(token)
