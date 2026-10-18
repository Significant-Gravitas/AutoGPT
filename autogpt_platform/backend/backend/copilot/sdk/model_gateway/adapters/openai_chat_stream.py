"""Chat deltas mapped to Anthropic blocks; tool fragments accumulate until complete."""

from pydantic import BaseModel

from backend.copilot.sdk.model_gateway.adapters.openai_tools import arguments_json
from backend.copilot.sdk.model_gateway.stream import AnthropicStreamBuilder, Event
from backend.copilot.sdk.model_gateway.tool_names import cli_tool_id
from backend.copilot.sdk.model_gateway.usage import GatewayUsage, usage_from_openai

_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


class _PendingCall(BaseModel):
    id: str | None = None
    name: str = ""
    arguments: str = ""


class ChatStreamMapper:
    def __init__(self, builder: AnthropicStreamBuilder, names: dict[str, str]) -> None:
        self._builder = builder
        self._names = names
        self._calls: dict[int, _PendingCall] = {}
        self._finish_reason: str | None = None
        self._usage: GatewayUsage | None = None
        self.saw_done = False

    @property
    def complete(self) -> bool:
        """Whether the server ended the response rather than dropping it."""
        return self.saw_done or self._finish_reason is not None

    @property
    def usage(self) -> GatewayUsage | None:
        return self._usage

    def feed(self, chunk: Event) -> list[Event]:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self._usage = chat_usage(usage)
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        choice = choices[0]
        if not isinstance(choice, dict):
            return []
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self._finish_reason = finish_reason
        delta = choice.get("delta")
        return self._delta(delta) if isinstance(delta, dict) else []

    def finish(self) -> list[Event]:
        events: list[Event] = []
        if self._finish_reason == "content_filter" and not self._builder.has_text:
            events += self._builder.text("The model server declined to answer.")
        for _, call in sorted(self._calls.items()):
            name = self._names.get(call.name, call.name)
            events += self._builder.tool_use(
                cli_tool_id(call.id), name, call.arguments or "{}"
            )
        stop = _STOP_REASONS.get(self._finish_reason or "stop", "end_turn")
        if self._builder.tool_used and stop == "end_turn":
            # Some servers say "stop" even when the turn ended in tool calls.
            stop = "tool_use"
        return events + self._builder.finish(stop, self._usage)

    def _delta(self, delta: dict[object, object]) -> list[Event]:
        events: list[Event] = []
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            events += self._builder.thinking(reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            events += self._builder.text(content)
        refusal = delta.get("refusal")
        if isinstance(refusal, str):
            events += self._builder.text(refusal)
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for position, call in enumerate(calls):
                if isinstance(call, dict):
                    self._collect_call(call, position)
        return events

    def _collect_call(self, call: dict[object, object], position: int) -> None:
        index = call.get("index")
        key = index if isinstance(index, int) else position
        pending = self._calls.setdefault(key, _PendingCall())
        if isinstance(call.get("id"), str) and call["id"]:
            pending.id = (pending.id or "") + str(call["id"])
        function = call.get("function")
        if not isinstance(function, dict):
            return
        if isinstance(function.get("name"), str) and function["name"]:
            pending.name += str(function["name"])
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            pending.arguments += arguments
        elif "arguments" in function:
            pending.arguments = arguments_json(arguments)


def chat_usage(raw: dict[object, object]) -> GatewayUsage:
    return usage_from_openai(raw, responses=False)
