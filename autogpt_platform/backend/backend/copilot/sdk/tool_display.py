"""Correlate SDK MCP executions with provider tool calls without matching arguments."""

import asyncio
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from backend.copilot.response_model import StreamToolDisplayAvailable, ToolDisplayData
from backend.copilot.tool_display import tool_display_context

if TYPE_CHECKING:
    from backend.copilot.model import ChatMessage

_TOKEN_KEY = "__agpt_display_token"


class _CallBinding(BaseModel):
    tool_name: str
    call_id: str
    generation: int


class SDKToolDisplayBridge:
    """Bind PreToolUse IDs to MCP handlers, which receive only arguments.

    A one-use token travels through the hook's updatedInput and is stripped
    before execution. This also distinguishes concurrent identical calls.
    Retry generations prevent callbacks from abandoned calls leaking forward.
    """

    def __init__(self) -> None:
        self.ready = asyncio.Event()
        self._generation = 0
        self._bindings: dict[str, _CallBinding] = {}
        self._pending: list[StreamToolDisplayAvailable] = []
        self._names: dict[str, str] = {}

    def prepare_call(
        self, tool_name: str, arguments: dict[str, Any], call_id: str
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex
        self._bindings[token] = _CallBinding(
            tool_name=tool_name, call_id=call_id, generation=self._generation
        )
        return {**strip_display_token(arguments), _TOKEN_KEY: token}

    @contextmanager
    def execution_context(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        token = arguments.get(_TOKEN_KEY)
        binding = self._bindings.pop(token, None) if isinstance(token, str) else None
        if binding is not None and binding.tool_name != tool_name:
            binding = None
        with tool_display_context(lambda name: self._publish(binding, name)):
            yield strip_display_token(arguments)

    def drain(self) -> list[StreamToolDisplayAvailable]:
        pending, self._pending = self._pending, []
        self.ready.clear()
        return pending

    def reset(self) -> None:
        self._generation += 1
        self._bindings.clear()
        self._pending.clear()
        self._names.clear()
        self.ready.clear()

    def _publish(self, binding: _CallBinding | None, name: str) -> None:
        if binding is None or binding.generation != self._generation:
            return
        if self._names.get(binding.call_id) == name:
            return
        self._names[binding.call_id] = name
        self._pending.append(
            StreamToolDisplayAvailable(
                id=binding.call_id,
                data=ToolDisplayData(toolCallId=binding.call_id, displayName=name),
            )
        )
        self.ready.set()


def strip_display_token(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key != _TOKEN_KEY}


def stamp_tool_display_name(
    messages: Iterable["ChatMessage"], call_id: str, display_name: str
) -> None:
    for message in messages:
        for call in message.tool_calls or []:
            if call.get("id") == call_id and call.get("display_name") != display_name:
                call["display_name"] = display_name
                message.mark_tool_calls_pending_save()
