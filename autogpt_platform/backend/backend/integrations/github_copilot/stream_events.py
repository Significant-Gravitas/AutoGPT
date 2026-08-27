"""Accumulating a GitHub Copilot turn from its JSON-RPC event stream.

The counterpart to ``microsoft_365_copilot.stream_parser``, and deliberately
*not* the same shape -- which is the whole reason both exist as named
modules rather than one "copilot parser". Microsoft streams cumulative
snapshots and you emit the new suffix. GitHub streams incremental deltas and
you append them. Using either strategy on the other producer is silently
wrong: appending Microsoft's snapshots repeats the answer once per event,
and diffing GitHub's deltas emits almost nothing.

The events arrive as JSON-RPC notifications (``session.event``) rather than
SSE, and every one shares an envelope::

    {id, timestamp, parentId, agentId?, ephemeral?, type, data}

Only a few types carry the answer:

``assistant.message_delta``
    Ephemeral, incremental. ``{messageId, deltaContent}`` -- append.
``assistant.message``
    The *complete* text for that call, same ``messageId``. Authoritative:
    it supersedes what the deltas built rather than adding to it, which is
    what makes a dropped or duplicated delta self-correcting.
``assistant.streaming_delta``
    A trap. It is cumulative *and* it is not text -- it carries
    ``totalResponseSizeBytes``, a network progress counter. Wiring it to a
    renderer puts byte counts on screen.
``session.idle``
    The turn is finished. Not ``assistant.turn_end``: an agentic turn can
    span several LLM calls, and ending on the first one truncates the
    answer mid-run.
``session.error``
    Carries ``errorType`` and ``statusCode``, which is what lets a failure
    say something a user can act on instead of "something went wrong".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DELTA = "assistant.message_delta"
_MESSAGE = "assistant.message"
_REASONING_DELTA = "assistant.reasoning_delta"
_IDLE = "session.idle"
_ERROR = "session.error"


@dataclass
class TurnError:
    """A failure with enough on it to say something useful.

    ``error_type`` is the provider's own discriminator ("authentication",
    "quota", "rate_limit", ...). Kept rather than flattened to a string,
    because "your Copilot allowance is used up" and "your token expired"
    need different words and different buttons.
    """

    error_type: str
    message: str
    status_code: int | None = None
    # GitHub's request id, echoed so a support conversation can find the
    # call. Useless to a user, decisive for anyone debugging one.
    provider_call_id: str | None = None


@dataclass
class CopilotTurn:
    """What has been accumulated for one turn so far."""

    text: str = ""
    reasoning: str = ""
    is_complete: bool = False
    was_aborted: bool = False
    error: TurnError | None = None
    _messages: dict[str, str] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)


class CopilotEventAccumulator:
    """Stateful across one turn. Not reusable for a second."""

    def __init__(self) -> None:
        self.turn = CopilotTurn()

    def feed(self, event: dict[str, Any]) -> str:
        """One event in, newly-arrived display text out.

        Returns "" for everything that adds no text -- progress counters,
        turn boundaries, reasoning -- so a caller can emit unconditionally
        without producing empty chunks.
        """
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}

        if event_type == _DELTA:
            return self._append_delta(data)
        if event_type == _MESSAGE:
            return self._replace_with_final(data)
        if event_type == _REASONING_DELTA:
            self.turn.reasoning += _text(data, "deltaContent", "content")
            return ""
        if event_type == _IDLE:
            self.turn.is_complete = True
            self.turn.was_aborted = bool(data.get("aborted"))
            return ""
        if event_type == _ERROR:
            self._record_error(data)
            return ""
        return ""

    def _append_delta(self, data: dict[str, Any]) -> str:
        message_id = str(data.get("messageId") or "")
        chunk = _text(data, "deltaContent", "content")
        if not chunk:
            return ""
        if message_id not in self.turn._messages:
            self.turn._order.append(message_id)
            self.turn._messages[message_id] = ""
        self.turn._messages[message_id] += chunk
        self._rebuild()
        return chunk

    def _replace_with_final(self, data: dict[str, Any]) -> str:
        """The complete text for one LLM call, which supersedes its deltas.

        Returning only the part not already shown keeps the stream from
        repeating itself on screen, while still letting the authoritative
        text win -- so a dropped delta is repaired here rather than leaving
        a hole in the answer nobody notices.
        """
        message_id = str(data.get("messageId") or data.get("id") or "")
        final = _text(data, "content", "text")
        if not final:
            return ""
        already = self.turn._messages.get(message_id, "")
        if message_id not in self.turn._messages:
            self.turn._order.append(message_id)
        self.turn._messages[message_id] = final
        self._rebuild()
        return final[len(already) :] if final.startswith(already) else final

    def _record_error(self, data: dict[str, Any]) -> None:
        status = data.get("statusCode")
        self.turn.error = TurnError(
            error_type=str(data.get("errorType") or "unknown"),
            message=str(data.get("message") or "The provider reported an error."),
            status_code=int(status) if isinstance(status, (int, float)) else None,
            provider_call_id=(
                str(data["providerCallId"]) if data.get("providerCallId") else None
            ),
        )
        # A turn that errored is over. Without this a caller waiting for
        # ``session.idle`` waits for a message that is not coming.
        self.turn.is_complete = True

    def _rebuild(self) -> None:
        # An agentic turn can span several LLM calls, so the answer is the
        # concatenation of its messages in arrival order -- keeping only the
        # latest would drop everything the model said before its last tool
        # call.
        self.turn.text = "".join(self.turn._messages[key] for key in self.turn._order)


def _text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""
