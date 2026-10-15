"""Anthropic token buckets, exact-usage accumulation and request-size estimates."""

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

# What an image costs in the estimate; base64 bytes say nothing about tokens.
_IMAGE_TOKENS = 1_600
_CHARS_PER_TOKEN = 4


class GatewayUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read, whichever bucket it was billed in."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def plus(self, other: "GatewayUsage") -> "GatewayUsage":
        return GatewayUsage(
            **{
                key: value + other.model_dump()[key]
                for key, value in self.model_dump().items()
            }
        )

    def anthropic(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_tokens,
            "cache_creation_input_tokens": self.cache_write_tokens,
        }


def usage_from_anthropic(raw: object) -> GatewayUsage | None:
    """Read an Anthropic ``usage`` object; ``None`` when there is none."""
    if not isinstance(raw, Mapping):
        return None
    names = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "cache_read_input_tokens": "cache_read_tokens",
        "cache_creation_input_tokens": "cache_write_tokens",
    }
    return GatewayUsage(
        **{field: count(raw[name]) for name, field in names.items() if name in raw}
    )


def merge_stream_usage(start: GatewayUsage, delta: GatewayUsage) -> GatewayUsage:
    """Reported fields replace earlier counts, including explicit zeros."""
    return GatewayUsage(
        **{
            **start.model_dump(exclude_unset=True),
            **delta.model_dump(exclude_unset=True),
        }
    )


def usage_from_openai(raw: Mapping[object, object], *, responses: bool) -> GatewayUsage:
    input_key = "input_tokens" if responses else "prompt_tokens"
    output_key = "output_tokens" if responses else "completion_tokens"
    details = raw.get(f"{input_key}_details")
    details = details if isinstance(details, dict) else {}
    output = raw.get(f"{output_key}_details")
    output = output if isinstance(output, dict) else {}
    cached = count(details.get("cached_tokens")) or count(
        raw.get("cache_read_input_tokens")
    )
    written = None if responses else _cache_write(details, raw)
    fields = {}
    if input_key in raw:
        fields["input_tokens"] = max(0, count(raw[input_key]) - cached - (written or 0))
    if output_key in raw:
        fields["output_tokens"] = count(raw[output_key])
    if "reasoning_tokens" in output:
        fields["reasoning_tokens"] = count(output["reasoning_tokens"])
    if "cached_tokens" in details or "cache_read_input_tokens" in raw:
        fields["cache_read_tokens"] = cached
    if written is not None:
        fields["cache_write_tokens"] = written
    return GatewayUsage(**fields)


def _cache_write(
    details: Mapping[object, object], raw: Mapping[object, object]
) -> int | None:
    names = (
        "cache_write_tokens",
        "cache_creation_tokens",
        "cache_creation_input_tokens",
    )
    return next(
        (
            count(source[name])
            for source in (details, raw)
            for name in names
            if name in source
        ),
        None,
    )


class UsageMeter:
    """The same accumulator for streaming, collection and failed attempts."""

    def __init__(self) -> None:
        self._usage = GatewayUsage()
        self._exact: GatewayUsage | None = None
        self.failed = False

    @property
    def total(self) -> GatewayUsage:
        return self._exact if self._exact is not None else self._usage

    def observe(
        self, event: Mapping[str, object], exact: GatewayUsage | None = None
    ) -> None:
        if event.get("type") == "message_start":
            message = event.get("message")
            if isinstance(message, dict):
                self.observe_message(message)
        elif event.get("type") == "message_delta":
            self.observe_message(event)
        if exact is not None:
            self._exact = merge_stream_usage(self._usage, exact)

    def observe_message(self, message: Mapping[str, object]) -> None:
        if (usage := usage_from_anthropic(message.get("usage"))) is not None:
            self._usage = merge_stream_usage(self._usage, usage)

    def fail(self, usage: GatewayUsage | None = None) -> None:
        self.failed = True
        if usage is not None:
            self._exact = usage


def estimate_tokens(body: Mapping[str, object]) -> int:
    """Fallback prompt size for final usage, count_tokens and overflow errors."""
    chars, images = _measure(
        [body.get("system"), body.get("messages"), body.get("tools")]
    )
    return max(1, chars // _CHARS_PER_TOKEN + images * _IMAGE_TOKENS)


def _measure(node: object) -> tuple[int, int]:
    if isinstance(node, str):
        return len(node), 0
    if isinstance(node, Mapping):
        if node.get("type") == "base64" and isinstance(node.get("data"), str):
            return 0, 1
        return _total(_measure(value) for value in node.values())
    if isinstance(node, list):
        return _total(_measure(value) for value in node)
    return 0, 0


def _total(pairs: Iterable[tuple[int, int]]) -> tuple[int, int]:
    chars = images = 0
    for pair_chars, pair_images in pairs:
        chars += pair_chars
        images += pair_images
    return chars, images


def count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
