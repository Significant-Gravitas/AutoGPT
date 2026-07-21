"""Stream event → aggregated result accumulator.

Consumes the same ``StreamBaseResponse`` events that fly over
``stream_registry`` (text deltas, tool i/o, usage, errors) and folds
them into a single :class:`EventAccumulator` state.  Used by
:func:`session_waiter.wait_for_session_result` to read events from a
Redis Stream subscription so a different process can obtain the
aggregated result for a session it didn't run.

Keeping the dispatch in one place means new event types can be added
without drifting callers apart on what "response_text", "tool_calls",
or token counts mean.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..response_model import (
    StreamError,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    StreamUsage,
)

logger = logging.getLogger(__name__)


class ToolCallEntry(BaseModel):
    """A single tool call observed during stream consumption."""

    tool_call_id: str
    tool_name: str
    input: Any
    output: Any = None
    success: bool | None = None


class EventAccumulator(BaseModel):
    """Mutable accumulator fed by :func:`process_event`."""

    response_parts: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallEntry] = Field(default_factory=list)
    tool_calls_by_id: dict[str, ToolCallEntry] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def process_event(event: object, acc: EventAccumulator) -> str | None:
    """Fold *event* into *acc*. Returns the error text on ``StreamError``.

    Uses structural pattern matching for dispatch per project guidelines.
    """
    match event:
        case StreamTextDelta(delta=delta):
            acc.response_parts.append(delta)
        case StreamToolInputAvailable() as e:
            entry = ToolCallEntry(
                tool_call_id=e.toolCallId,
                tool_name=e.toolName,
                input=e.input,
            )
            acc.tool_calls.append(entry)
            acc.tool_calls_by_id[e.toolCallId] = entry
        case StreamToolOutputAvailable() as e:
            if tc := acc.tool_calls_by_id.get(e.toolCallId):
                tc.output = e.output
                tc.success = e.success
            else:
                logger.debug(
                    "Received tool output for unknown tool_call_id: %s",
                    e.toolCallId,
                )
        case StreamUsage() as e:
            acc.prompt_tokens += e.prompt_tokens
            acc.completion_tokens += e.completion_tokens
            acc.total_tokens += e.total_tokens
        case StreamError(errorText=err):
            return err
    return None
