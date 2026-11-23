"""How long each tool call took, kept from the moment the tool returns until
its result is saved with the chat.

The tool runs in one place (``BaseTool.execute``) and its result message is
written in another (each engine's own bookkeeping), so the duration waits
here, keyed by tool call id. Bounded: a result that is never saved is dropped
once newer ones push it out.
"""

from collections import OrderedDict

_MAX_REMEMBERED = 2_000
_durations: OrderedDict[str, int] = OrderedDict()


def note(tool_call_id: str, milliseconds: float) -> None:
    """Remember how long the call took."""
    if not tool_call_id:
        return
    _durations[tool_call_id] = max(0, round(milliseconds))
    _durations.move_to_end(tool_call_id)
    while len(_durations) > _MAX_REMEMBERED:
        _durations.popitem(last=False)


def take(tool_call_id: str | None) -> int | None:
    """The call's duration in milliseconds, handed over once; None when the
    tool ran elsewhere or was never timed."""
    if not tool_call_id:
        return None
    return _durations.pop(tool_call_id, None)
