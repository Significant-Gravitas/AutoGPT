"""What the fake provider bills for each call it serves.

The prices are written out here instead of read from ``anthropic_rates.json``,
the LLM catalog or ``moonshot.py``. An invoice priced by the code under test
could not catch that code pricing a call wrongly.
"""

from typing import Literal

from pydantic import BaseModel

Route = Literal["anthropic", "openrouter", "ollama", "codex", "graph"]
CacheTtl = Literal["5m", "1h"]

SONNET = "claude-sonnet-5-5"
KIMI = "moonshotai/kimi-k2.6"


class Usage(BaseModel):
    """Token counts for one call; ``input_tokens`` excludes the cache buckets."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Price(BaseModel):
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float

    def of(self, usage: Usage, ttl: CacheTtl | None) -> float:
        write = self.cache_write_1h if ttl == "1h" else self.cache_write_5m
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * write
        ) / 1_000_000


# Anthropic's list prices. The bundled CLI prices its calls from its own copy.
ANTHROPIC_PRICES: dict[str, Price] = {
    SONNET: Price(
        input=2.0, output=10.0, cache_read=0.2, cache_write_5m=2.5, cache_write_1h=4.0
    ),
}

# OpenRouter's prices, a tenth above Anthropic's list on purpose: a turn charged
# an estimate where it should have been charged this invoice shows up as a
# mismatch instead of matching by coincidence.
OPENROUTER_PRICES: dict[str, Price] = {
    f"anthropic/{SONNET}": Price(
        input=2.2, output=11.0, cache_read=0.22, cache_write_5m=2.75, cache_write_1h=4.4
    ),
    KIMI: Price(
        input=0.66,
        output=3.08,
        cache_read=0.165,
        cache_write_5m=0.66,
        cache_write_1h=0.66,
    ),
}

# The rate card the platform falls back to for Kimi K2.6 when it has no
# invoice: cache tokens count as input. Mirrors ``moonshot.py``'s documented
# default rather than importing it.
MOONSHOT_FALLBACK = Price(
    input=0.60, output=2.80, cache_read=0.60, cache_write_5m=0.60, cache_write_1h=0.60
)


class Call(BaseModel):
    """One request the provider served."""

    turn: int
    # The turn's model request it answered, numbered like PlannedFault.request.
    request: int
    route: Route
    model: str
    usage: Usage
    cache_ttl: CacheTtl | None
    invoice_usd: float
    generation_id: str | None = None


def scripted_usage(index: int, *, cached: bool) -> Usage:
    """Usage for the provider's ``index``-th call.

    Distinct for every call, so a charge carried over from another call or
    turn cannot match by coincidence. A request that asked for no caching is
    billed the same tokens as fresh input.
    """
    fresh = 1_200 + 37 * index
    read = 40_000 + 1_500 * index
    write = 3_000 + 250 * index
    output = 150 + 11 * index
    if not cached:
        return Usage(input_tokens=fresh + read + write, output_tokens=output)
    return Usage(
        input_tokens=fresh,
        output_tokens=output,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )


def price_table(route: Route) -> dict[str, Price] | None:
    """The table ``route`` bills from; None where nobody bills the platform."""
    if route == "anthropic":
        return ANTHROPIC_PRICES
    if route == "openrouter":
        return OPENROUTER_PRICES
    # Ollama is self-hosted; Codex and Microsoft 365 bill the user's own plan.
    return None
