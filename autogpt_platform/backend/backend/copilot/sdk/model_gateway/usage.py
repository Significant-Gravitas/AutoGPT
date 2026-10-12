"""Token usage in Anthropic's buckets, and the request-size estimate.

Anthropic splits the prompt into uncached input, cache reads and cache writes;
OpenAI-style upstreams report one prompt count with the cached share inside
it. Everything here is held in Anthropic's terms, because the CLI reads it.
"""

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
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
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
    return GatewayUsage(
        input_tokens=_count(raw.get("input_tokens")),
        cache_read_tokens=_count(raw.get("cache_read_input_tokens")),
        cache_write_tokens=_count(raw.get("cache_creation_input_tokens")),
        output_tokens=_count(raw.get("output_tokens")),
    )


def merge_stream_usage(start: GatewayUsage, delta: GatewayUsage) -> GatewayUsage:
    """Combine ``message_start`` and ``message_delta`` usage as the CLI does.

    A positive count in the delta replaces the start's; output always comes
    from the delta.
    """
    return GatewayUsage(
        input_tokens=delta.input_tokens or start.input_tokens,
        cache_read_tokens=delta.cache_read_tokens or start.cache_read_tokens,
        cache_write_tokens=delta.cache_write_tokens or start.cache_write_tokens,
        output_tokens=delta.output_tokens,
        reasoning_tokens=delta.reasoning_tokens or start.reasoning_tokens,
    )


def estimate_tokens(body: Mapping[str, object]) -> int:
    """A rough count of what the request will cost the model to read.

    Characters over four, with a flat price per image. Used where nothing
    better exists: ``count_tokens``, the provisional figure in
    ``message_start``, and the prompt size in a rewritten over-long error
    when the upstream did not state one.
    """
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


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
