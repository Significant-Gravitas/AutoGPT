"""Codex Responses stream events, restated as Anthropic stream events.

Output items map one to one, in the order the model produced them:

- a ``reasoning`` item becomes a thinking block: its summary text streams in
  as thinking, and when the item is done its ``encrypted_content`` is sealed
  into the block's signature (see ``codex_signature``);
- a ``message`` item becomes a text block;
- a ``function_call`` item becomes a ``tool_use`` block, emitted whole when
  the item is done.

``response.completed``/``response.incomplete`` carry the usage and the stop
reason. ``response.failed`` and ``error`` are recorded as ``failure`` for the
adapter to raise; before any output item they still become an HTTP status.
"""

from backend.copilot.sdk.model_gateway.adapters.codex_signature import seal
from backend.copilot.sdk.model_gateway.errors import UpstreamError, error_from_payload
from backend.copilot.sdk.model_gateway.stream import AnthropicStreamBuilder, Event
from backend.copilot.sdk.model_gateway.tool_names import cli_tool_id
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

_INCOMPLETE_STOP_REASONS = {
    "max_output_tokens": "max_tokens",
    "content_filter": "refusal",
}


class CodexStreamMapper:
    def __init__(
        self, builder: AnthropicStreamBuilder, names: dict[str, str], tag: str
    ) -> None:
        self._builder = builder
        self._names = names
        self._tag = tag
        self._streamed_text: set[str] = set()
        self._usage: GatewayUsage | None = None
        self._stop: str | None = None
        self.completed = False
        self.failure: UpstreamError | None = None

    @property
    def usage(self) -> GatewayUsage | None:
        return self._usage

    def feed(self, event: Event) -> list[Event]:
        kind = event.get("type")
        if kind == "response.output_item.added":
            return self._item_added(event.get("item"))
        if kind == "response.reasoning_summary_part.added":
            index = event.get("summary_index")
            separator = "\n\n" if isinstance(index, int) and index > 0 else ""
            return self._builder.thinking(separator)
        if kind == "response.reasoning_summary_text.delta":
            return self._builder.thinking(str(event.get("delta") or ""))
        if kind == "response.output_text.delta":
            self._streamed_text.add(str(event.get("item_id") or ""))
            return self._builder.text(str(event.get("delta") or ""))
        if kind == "response.output_item.done":
            return self._item_done(event.get("item"))
        if kind in ("response.completed", "response.incomplete"):
            self._terminal(event.get("response"))
        elif kind in ("response.failed", "error"):
            self.failure = _failure_of(event)
        return []

    def finish(self) -> list[Event]:
        stop = self._stop or "end_turn"
        if self._builder.tool_used and stop == "end_turn":
            stop = "tool_use"
        return self._builder.finish(stop, self._usage)

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
            events: list[Event] = []
            if str(item.get("id") or "") not in self._streamed_text:
                events += self._builder.text(_message_text(item))
            return events + self._builder.close_block()
        if kind == "function_call":
            name = str(item.get("name") or "")
            return self._builder.tool_use(
                cli_tool_id(_text(item.get("call_id"))),
                self._names.get(name, name),
                _text(item.get("arguments")) or "{}",
            )
        return []

    def _terminal(self, response: object) -> None:
        self.completed = True
        if not isinstance(response, dict):
            return
        usage = response.get("usage")
        if isinstance(usage, dict):
            self._usage = responses_usage(usage)
        details = response.get("incomplete_details")
        if isinstance(details, dict):
            reason = str(details.get("reason") or "")
            self._stop = _INCOMPLETE_STOP_REASONS.get(reason, "end_turn")


def responses_usage(raw: dict[object, object]) -> GatewayUsage:
    """Responses usage in Anthropic's buckets: ``input_tokens`` includes the
    cached share, which Anthropic counts separately."""
    input_details = raw.get("input_tokens_details")
    cached = (
        _count(input_details.get("cached_tokens"))
        if isinstance(input_details, dict)
        else 0
    )
    output_details = raw.get("output_tokens_details")
    reasoning = (
        _count(output_details.get("reasoning_tokens"))
        if isinstance(output_details, dict)
        else 0
    )
    return GatewayUsage(
        input_tokens=max(0, _count(raw.get("input_tokens")) - cached),
        cache_read_tokens=cached,
        output_tokens=_count(raw.get("output_tokens")),
        reasoning_tokens=reasoning,
    )


def _failure_of(event: Event) -> UpstreamError:
    response = event.get("response")
    source = response if isinstance(response, dict) else event
    return error_from_payload(source) or UpstreamError(
        message="ChatGPT reported the response as failed"
    )


def _message_text(item: dict[object, object]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "output_text"
    )


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
