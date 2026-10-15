"""Anthropic events and collection, with one usage accumulator for both modes."""

import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.usage import (
    GatewayUsage,
    UsageMeter,
    merge_stream_usage,
)

Event = dict[str, object]
BlockKind = Literal["text", "thinking", "tool_use"]


class StreamItem(BaseModel):
    """One event for the CLI.

    ``raw`` is the exact frame to relay, when there is one. ``usage`` is the
    upstream's own count on the closing ``message_delta``, with what Anthropic's
    usage object has no field for (reasoning tokens).
    """

    model_config = ConfigDict(frozen=True)

    event: Event
    raw: bytes | None = None
    usage: GatewayUsage | None = None


def closing_items(events: list[Event], usage: GatewayUsage | None) -> list[StreamItem]:
    """The final events of a response, the exact usage riding on its delta."""
    return [
        StreamItem(
            event=event, usage=usage if event.get("type") == "message_delta" else None
        )
        for event in events
    ]


class Upstream(Protocol):
    """One upstream adapter, for one gateway's lifetime."""

    def stream(self, body: Event) -> AsyncGenerator[StreamItem, None]:
        """Anthropic events for ``body``; a ``GatewayError`` before the first
        one becomes the HTTP response, after it an SSE ``error`` event."""
        ...

    async def complete(self, body: Event) -> Event:
        """The whole Anthropic message for a non-streaming request."""
        return await collect_message(self.stream(body))

    async def aclose(self) -> None: ...


class AnthropicStreamBuilder:
    """Anthropic stream events for one upstream response, built incrementally."""

    def __init__(self, *, model: str, input_estimate: int) -> None:
        self._model = model
        self._estimate = input_estimate
        self._started = False
        self._index = -1
        self._open: BlockKind | None = None
        self.tool_used = False
        self.has_text = False

    def start(self) -> list[Event]:
        if self._started:
            return []
        self._started = True
        usage = GatewayUsage().anthropic()
        message: Event = {
            "id": f"msg_gateway_{uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage,
        }
        return [{"type": "message_start", "message": message}]

    def text(self, delta: str) -> list[Event]:
        if not delta:
            return []
        self.has_text = True
        events = self._ensure_open("text", {"type": "text", "text": ""})
        return events + [self._delta({"type": "text_delta", "text": delta})]

    def thinking(self, delta: str) -> list[Event]:
        events = self._ensure_open("thinking", _THINKING_START)
        if delta:
            events.append(self._delta({"type": "thinking_delta", "thinking": delta}))
        return events

    def signature(self, signature: str) -> list[Event]:
        """Seal the open thinking block (opening an empty one if needed)."""
        events = self._ensure_open("thinking", _THINKING_START)
        events.append(self._delta({"type": "signature_delta", "signature": signature}))
        return events + self.close_block()

    def tool_use(self, call_id: str, name: str, arguments: str) -> list[Event]:
        """A whole ``tool_use`` block: start, its JSON input, stop."""
        tool_arguments(arguments)
        self.tool_used = True
        block = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
        events = self.close_block() + self._open_block("tool_use", block)
        events.append(
            self._delta({"type": "input_json_delta", "partial_json": arguments})
        )
        return events + self.close_block()

    def close_block(self) -> list[Event]:
        if self._open is None:
            return []
        self._open = None
        return [{"type": "content_block_stop", "index": self._index}]

    def finish(self, stop_reason: str, usage: GatewayUsage | None) -> list[Event]:
        reported: Event = {
            **merge_stream_usage(
                GatewayUsage(input_tokens=self._estimate), usage or GatewayUsage()
            ).anthropic()
        }
        return [
            *self.start(),
            *self.close_block(),
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": reported,
            },
            {"type": "message_stop"},
        ]

    def _ensure_open(self, kind: BlockKind, block: Event) -> list[Event]:
        if self._open == kind:
            return []
        return self.close_block() + self._open_block(kind, block)

    def _open_block(self, kind: BlockKind, block: Event) -> list[Event]:
        events = self.start()
        self._index += 1
        self._open = kind
        events.append(
            {
                "type": "content_block_start",
                "index": self._index,
                "content_block": dict(block),
            }
        )
        return events

    def _delta(self, delta: Event) -> Event:
        return {"type": "content_block_delta", "index": self._index, "delta": delta}


_THINKING_START: Event = {"type": "thinking", "thinking": "", "signature": ""}


async def collect_message(
    items: AsyncIterable[StreamItem], meter: UsageMeter | None = None
) -> Event:
    """Fold a stream into the message a non-streaming request returns."""
    message: Event = {}
    blocks: list[Event] = []
    partial: dict[int, str] = {}
    meter = meter if meter is not None else UsageMeter()
    async for item in items:
        meter.observe(item.event, item.usage)
        event = item.event
        kind = event.get("type")
        started, block = event.get("message"), event.get("content_block")
        if kind == "message_start" and isinstance(started, dict):
            message = dict(started)
        elif kind == "content_block_start" and isinstance(block, dict):
            blocks.append(dict(block))
        elif kind == "content_block_delta" and blocks:
            _apply_delta(blocks[-1], event.get("delta"), partial, len(blocks) - 1)
        elif kind == "content_block_stop" and blocks:
            _finish_block(blocks[-1], partial.pop(len(blocks) - 1, None))
        elif kind == "message_delta":
            _apply_message_delta(message, event)
    message["content"] = blocks
    message["usage"] = meter.total.anthropic()
    return message


def _apply_delta(
    block: Event, delta: object, partial: dict[int, str], index: int
) -> None:
    if not isinstance(delta, dict):
        return
    kind = delta.get("type")
    if kind == "text_delta":
        block["text"] = f"{block.get('text', '')}{delta.get('text', '')}"
    elif kind == "thinking_delta":
        block["thinking"] = f"{block.get('thinking', '')}{delta.get('thinking', '')}"
    elif kind == "signature_delta":
        block["signature"] = delta.get("signature", "")
    elif kind == "input_json_delta":
        partial[index] = partial.get(index, "") + str(delta.get("partial_json", ""))


def _finish_block(block: Event, arguments: str | None) -> None:
    if block.get("type") != "tool_use" or arguments is None:
        return
    block["input"] = tool_arguments(arguments)


def tool_arguments(arguments: str) -> dict[str, object]:
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except ValueError:
        raise GatewayError(
            502, "api_error", "The model server returned invalid tool arguments"
        ) from None
    if not isinstance(parsed, dict):
        raise GatewayError(
            502, "api_error", "The model server returned non-object tool arguments"
        )
    return parsed


def _apply_message_delta(message: Event, event: Event) -> None:
    delta = event.get("delta")
    if isinstance(delta, dict):
        message["stop_reason"] = delta.get("stop_reason")
        message["stop_sequence"] = delta.get("stop_sequence")
