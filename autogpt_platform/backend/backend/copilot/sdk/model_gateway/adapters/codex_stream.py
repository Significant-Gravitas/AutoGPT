"""Responses events mapped to Anthropic text, thinking and whole tool blocks.

Completed reasoning is authenticated before the CLI carries it in signatures.
Terminal usage is retained even when the response fails."""

from backend.copilot.sdk.model_gateway.adapters.codex_signature import (
    ReasoningIdentity,
    seal,
)
from backend.copilot.sdk.model_gateway.adapters.openai_tools import arguments_json
from backend.copilot.sdk.model_gateway.errors import UpstreamError, error_from_payload
from backend.copilot.sdk.model_gateway.stream import AnthropicStreamBuilder, Event
from backend.copilot.sdk.model_gateway.tool_names import cli_tool_id
from backend.copilot.sdk.model_gateway.usage import (
    GatewayUsage,
    count,
    merge_stream_usage,
    usage_from_openai,
)

_INCOMPLETE_STOP_REASONS = {
    "max_output_tokens": "max_tokens",
    "content_filter": "end_turn",
}


class CodexStreamMapper:
    def __init__(
        self,
        builder: AnthropicStreamBuilder,
        names: dict[str, str],
        tag: ReasoningIdentity,
    ) -> None:
        self._builder = builder
        self._names = names
        self._tag = tag
        self._streamed_text: set[tuple[str, int]] = set()
        self._usage: GatewayUsage | None = None
        self._stop: str | None = None
        self._filtered = False
        self.completed = False
        self.failure: UpstreamError | None = None

    @property
    def usage(self) -> GatewayUsage | None:
        return self._usage

    def feed(self, event: Event) -> list[Event]:
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            incoming = responses_usage(response["usage"])
            self._usage = (
                merge_stream_usage(self._usage, incoming)
                if self._usage is not None
                else incoming
            )
        kind = event.get("type")
        if kind == "response.output_item.added":
            return self._item_added(event.get("item"))
        if kind == "response.reasoning_summary_part.added":
            index = event.get("summary_index")
            separator = "\n\n" if isinstance(index, int) and index > 0 else ""
            return self._builder.thinking(separator)
        if kind == "response.reasoning_summary_text.delta":
            return self._builder.thinking(str(event.get("delta") or ""))
        if kind in ("response.output_text.delta", "response.refusal.delta"):
            self._streamed_text.add(_part_key(event))
            return self._builder.text(str(event.get("delta") or ""))
        if kind in ("response.refusal.done", "response.content_part.done"):
            return self._part_done(event)
        if kind == "response.output_item.done":
            return self._item_done(event.get("item"))
        if kind in ("response.completed", "response.incomplete"):
            self._terminal(event.get("response"))
        elif kind in ("response.failed", "error"):
            self._terminal(event.get("response"))
            self.failure = _failure_of(event)
        return []

    def finish(self) -> list[Event]:
        events = (
            self._builder.text("The model server declined to answer.")
            if self._filtered and not self._builder.has_text
            else []
        )
        stop = self._stop or "end_turn"
        if self._builder.tool_used and stop == "end_turn":
            stop = "tool_use"
        return events + self._builder.finish(stop, self._usage)

    def _item_added(self, item: object) -> list[Event]:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            return self._builder.thinking("")
        return self._builder.start()

    def _item_done(self, item: object) -> list[Event]:
        if not isinstance(item, dict):
            return []
        kind = item.get("type")
        if kind == "reasoning":
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                return self._builder.signature(seal(encrypted, self._tag))
            return self._builder.close_block()
        if kind == "message":
            content = item.get("content")
            events = []
            if isinstance(content, list):
                for index, part in enumerate(content):
                    key = (str(item.get("id") or ""), index)
                    if key not in self._streamed_text:
                        events += self._builder.text(_part_text(part))
                        self._streamed_text.add(key)
            return events + self._builder.close_block()
        if kind == "function_call":
            name = str(item.get("name") or "")
            return self._builder.tool_use(
                cli_tool_id(_text(item.get("call_id"))),
                self._names.get(name, name),
                arguments_json(item.get("arguments", {})),
            )
        return []

    def _part_done(self, event: Event) -> list[Event]:
        key = _part_key(event)
        if key in self._streamed_text:
            return []
        text = (
            _part_text(event.get("part"))
            if event.get("type") == "response.content_part.done"
            else str(event.get("refusal") or "")
        )
        self._streamed_text.add(key)
        return self._builder.text(text)

    def _terminal(self, response: object) -> None:
        self.completed = True
        if not isinstance(response, dict):
            return
        details = response.get("incomplete_details")
        if isinstance(details, dict):
            reason = str(details.get("reason") or "")
            self._filtered = reason == "content_filter"
            self._stop = _INCOMPLETE_STOP_REASONS.get(reason, "end_turn")


def responses_usage(raw: dict[object, object]) -> GatewayUsage:
    return usage_from_openai(raw, responses=True)


def _failure_of(event: Event) -> UpstreamError:
    response = event.get("response")
    source = response if isinstance(response, dict) else event
    return error_from_payload(source) or UpstreamError(
        message="ChatGPT reported the response as failed"
    )


def _part_text(part: object) -> str:
    if not isinstance(part, dict):
        return ""
    if part.get("type") == "refusal":
        return str(part.get("refusal") or "")
    return str(part.get("text") or "") if part.get("type") == "output_text" else ""


def _part_key(event: Event) -> tuple[str, int]:
    return str(event.get("item_id") or ""), count(event.get("content_index"))


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
