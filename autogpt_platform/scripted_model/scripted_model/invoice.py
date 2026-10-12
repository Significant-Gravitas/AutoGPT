"""What each model call costs: the fake provider's own invoice.

The baseline engine is charged the ``usage.cost`` an OpenRouter-style
response carries, so the fake reports one. The SDK engine is charged the
CLI's estimate from its own copy of Anthropic's list prices, which these
match (backend/copilot/anthropic_rates.json). A journey checks what the
platform recorded against the sum of these, read from the journal.
"""

from dataclasses import dataclass
from typing import Any

from .scripts import Usage


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float


# Keyed by model name without the vendor prefix (``anthropic/``).
PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(2.0, 10.0, 0.2, 2.5, 4.0),
    "claude-opus-5-5": Price(4.0, 20.0, 0.2, 5.0, 8.0),
    "claude-haiku-4-5": Price(1.0, 5.0, 0.1, 1.25, 2.0),
}


def cost_usd(model: str, usage: Usage, cache_ttl: str | None) -> float | None:
    """The call's price, or None for a model this table doesn't list."""
    price = PRICES.get(model.rsplit("/", 1)[-1])
    if price is None:
        return None
    write = price.cache_write_1h if cache_ttl == "1h" else price.cache_write_5m
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * write
    ) / 1_000_000


def requested_cache_ttl(body: Any) -> str | None:
    """The prompt-cache lifetime a request asked for: "1h" if any marker asks
    for an hour, "5m" for any other marker, None when nothing is cached."""
    ttls = set(_cache_ttls(body))
    if not ttls:
        return None
    return "1h" if "1h" in ttls else "5m"


def _cache_ttls(node: Any):
    if isinstance(node, dict):
        control = node.get("cache_control")
        if isinstance(control, dict):
            yield str(control.get("ttl") or "5m")
        for value in node.values():
            yield from _cache_ttls(value)
    elif isinstance(node, list):
        for value in node:
            yield from _cache_ttls(value)
