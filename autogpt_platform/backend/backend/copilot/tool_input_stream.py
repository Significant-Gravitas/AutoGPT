"""Stream a tool call's arguments to the client while the model writes them.

Both engines open a call's tool part (``tool-input-start``) as soon as its name
is known, relay its argument JSON as ``tool-input-delta`` chunks, and close it
with ``tool-input-available`` once the arguments are complete: the AI SDK's own
sequence, so the card appears when the model starts the call rather than after
it has written every argument.

Deltas are relayed through Redis and replayed on reconnect, so they are batched
(``_FLUSH_CHARS`` / ``_FLUSH_INTERVAL_S``) and capped per call
(``_MAX_STREAMED_CHARS``, for file writes); ``tool-input-available`` carries the
whole arguments whatever the cap cut.

A ``run_capability`` call is held back: the tool it runs, and so the name its
part must carry, comes from its arguments (``resolve_tool_dispatch``), and the
AI SDK cannot rename a part once it has started. Its start goes out with its
input, as it always has.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from backend.copilot.capabilities.dispatch import DISPATCHER_TOOL_NAME
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamToolInputAvailable,
    StreamToolInputDelta,
    StreamToolInputError,
    StreamToolInputStart,
)

logger = logging.getLogger(__name__)

_FLUSH_CHARS = 200
_FLUSH_INTERVAL_S = 0.05
_MAX_STREAMED_CHARS = 20_000

INCOMPLETE_INPUT_ERROR = "The tool call was cut off before its input was complete."


class _OpenCall(BaseModel):
    tool_name: str
    last_flush: float
    pending: str = ""
    sent_chars: int = 0


class ToolInputStreams:
    """One turn's tool parts that have started and not yet received their input."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._open: dict[str, _OpenCall] = {}

    @property
    def has_open(self) -> bool:
        return bool(self._open)

    def start(self, call_id: str, tool_name: str) -> list[StreamBaseResponse]:
        """Open the call's part, unless it is open or its name is not final yet."""
        if not call_id or not tool_name or call_id in self._open:
            return []
        if tool_name == DISPATCHER_TOOL_NAME:
            return []
        self._open[call_id] = _OpenCall(tool_name=tool_name, last_flush=self._clock())
        return [StreamToolInputStart(toolCallId=call_id, toolName=tool_name)]

    def append(self, call_id: str, text: str) -> list[StreamBaseResponse]:
        """Buffer argument text; a delta goes out once enough has built up."""
        call = self._open.get(call_id)
        if call is None or not text:
            return []
        room = _MAX_STREAMED_CHARS - call.sent_chars - len(call.pending)
        if room <= 0:
            return []
        call.pending += text[:room]
        now = self._clock()
        if (
            len(call.pending) < _FLUSH_CHARS
            and now - call.last_flush < _FLUSH_INTERVAL_S
            and len(text) < room
        ):
            return []
        call.last_flush = now
        return _drain(call_id, call)

    def flush(self) -> list[StreamBaseResponse]:
        """Send every open call's buffered text, keeping the calls open."""
        return [
            event for cid, call in self._open.items() for event in _drain(cid, call)
        ]

    def complete(
        self, call_id: str, tool_name: str, tool_input: dict[str, Any]
    ) -> list[StreamBaseResponse]:
        """Close the call's part with its input, opening it first if nothing did."""
        call = self._open.pop(call_id, None)
        if call is None:
            events: list[StreamBaseResponse] = [
                StreamToolInputStart(toolCallId=call_id, toolName=tool_name)
            ]
        else:
            events = _drain(call_id, call)
            if call.tool_name != tool_name:
                logger.warning(
                    f"Tool part {call_id} opened as {call.tool_name} "
                    f"but its input names {tool_name}"
                )
        events.append(
            StreamToolInputAvailable(
                toolCallId=call_id, toolName=tool_name, input=tool_input
            )
        )
        return events

    def settle(self, call_id: str, tool_name: str) -> list[StreamBaseResponse]:
        """Ready the call's part for an output that ends it without an input.

        A refused call goes straight to its output, which the AI SDK can only
        attach to a part that exists, so one is opened when none is.
        """
        call = self._open.pop(call_id, None)
        if call is None:
            return [StreamToolInputStart(toolCallId=call_id, toolName=tool_name)]
        return _drain(call_id, call)

    def abandon(self) -> list[StreamBaseResponse]:
        """End every open part: its arguments never completed, so it never ran."""
        events: list[StreamBaseResponse] = [
            StreamToolInputError(
                toolCallId=call_id,
                toolName=call.tool_name,
                errorText=INCOMPLETE_INPUT_ERROR,
            )
            for call_id, call in self._open.items()
        ]
        self._open.clear()
        return events


def _drain(call_id: str, call: _OpenCall) -> list[StreamBaseResponse]:
    if not call.pending:
        return []
    delta = StreamToolInputDelta(toolCallId=call_id, inputTextDelta=call.pending)
    call.sent_chars += len(call.pending)
    call.pending = ""
    return [delta]
