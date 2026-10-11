"""Live spend for a running turn, priced call by call.

The CLI reports what a turn cost only once it ends. Landing a task before
the money runs out needs a running figure, so each model call the CLI
streams is priced as it arrives, with the same rate cards the post-turn
estimate uses.

``AssistantMessage`` is the single source. It carries the call's id and
usage for the main agent and for foreground sub-agents alike, and it repeats
once per content block, so calls are keyed by id and each count is kept at
its highest value. The figure runs slightly low -- a call's last block can
predate its final output count, and compaction calls aren't reported --
which the governor's reserve and the CLI backstop absorb. Billing still
comes from the post-turn reconcile; this only decides when to land.
"""

from functools import cache
from typing import Any

from claude_agent_sdk import AssistantMessage
from pydantic import BaseModel

from backend.copilot.anthropic_rate_card import compute_anthropic_cost_usd
from backend.copilot.moonshot import rate_card_usd
from backend.data.llm_registry.catalog import get_catalog

# The CLI's price table only knows Anthropic models and prices anything else
# like Sonnet. Also our price for a model no rate card names: overestimating
# lands a task early, underestimating overspends.
REFERENCE_MODEL = "claude-sonnet-4-5"

_PER_MTOK = 1_000_000


class CallUsage(BaseModel):
    """One model call's token counts, as far as the stream has reported."""

    model: str | None = None
    main: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def merge(self, usage: dict[str, Any]) -> None:
        self.input_tokens = max(self.input_tokens, _count(usage, "input_tokens"))
        self.output_tokens = max(self.output_tokens, _count(usage, "output_tokens"))
        self.cache_read_tokens = max(
            self.cache_read_tokens, _count(usage, "cache_read_input_tokens")
        )
        self.cache_creation_tokens = max(
            self.cache_creation_tokens, _count(usage, "cache_creation_input_tokens")
        )

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


class UsageMeter:
    """Spend and model-call count for one turn, across its retries."""

    def __init__(self) -> None:
        self._calls: dict[str, CallUsage] = {}
        self._priced: dict[str, float] = {}
        self._last_main_id: str | None = None

    def observe(self, message: AssistantMessage) -> None:
        call_id = message.message_id
        if not call_id or not message.usage:
            return
        call = self._calls.get(call_id)
        if call is None:
            call = CallUsage(
                model=message.model or None,
                main=message.parent_tool_use_id is None,
            )
            self._calls[call_id] = call
        call.merge(message.usage)
        self._priced[call_id] = price_call_usd(call)
        if call.main:
            self._last_main_id = call_id

    @property
    def spent_usd(self) -> float:
        return sum(self._priced.values())

    @property
    def main_calls(self) -> int:
        """Model calls made by the main agent: the CLI's ``max_turns`` unit."""
        return sum(1 for call in self._calls.values() if call.main)

    @property
    def last_main_call_usd(self) -> float:
        if self._last_main_id is None:
            return 0.0
        return self._priced[self._last_main_id]


def price_call_usd(call: CallUsage) -> float:
    """USD for one call at the model's list price."""
    anthropic = compute_anthropic_cost_usd(
        model=call.model or REFERENCE_MODEL,
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.output_tokens,
        cache_read_tokens=call.cache_read_tokens,
        cache_creation_tokens=call.cache_creation_tokens,
    )
    if anthropic is not None:
        return anthropic
    rates = _usd_per_mtok(call.model)
    if rates is None:
        return _reference_cost_usd(call)
    input_rate, output_rate = rates
    return (call.prompt_tokens * input_rate + call.output_tokens * output_rate) / (
        _PER_MTOK
    )


def cli_price_ratio(model: str | None) -> float:
    """How many of the CLI's budget dollars one real dollar on *model* is.

    The CLI prices non-Anthropic models like Sonnet, so its budget counter
    runs several times faster than the real spend on, say, Kimi (~5x). A
    backstop handed to the CLI has to be scaled up by this much or it stops
    the turn before our own landing does. The larger of the input and
    output ratios is used so the backstop never fires early.
    """
    if _is_anthropic(model):
        return 1.0
    rates = _usd_per_mtok(model)
    if rates is None:
        return 1.0
    reference_in, reference_out = _reference_rates()
    real_in, real_out = rates
    ratios = [
        reference / real
        for reference, real in ((reference_in, real_in), (reference_out, real_out))
        if real > 0
    ]
    return max([1.0, *ratios])


def _is_anthropic(model: str | None) -> bool:
    # The rate card prices Anthropic slugs and returns None for anything else.
    return (
        compute_anthropic_cost_usd(
            model=model or REFERENCE_MODEL, prompt_tokens=0, completion_tokens=0
        )
        is not None
    )


def _usd_per_mtok(model: str | None) -> tuple[float, float] | None:
    """(input, output) list price for a model priced outside Anthropic's card."""
    if not model:
        return None
    moonshot = rate_card_usd(model)
    if moonshot is not None:
        return moonshot
    return _catalog_rates().get(model)


@cache
def _catalog_rates() -> dict[str, tuple[float, float]]:
    return {
        m.slug: (m.cost.provider_input_usd_per_1m, m.cost.provider_output_usd_per_1m)
        for m in get_catalog().models
        if m.cost is not None
        and m.cost.provider_input_usd_per_1m is not None
        and m.cost.provider_output_usd_per_1m is not None
    }


@cache
def _reference_rates() -> tuple[float, float]:
    per_input = compute_anthropic_cost_usd(
        model=REFERENCE_MODEL, prompt_tokens=_PER_MTOK, completion_tokens=0
    )
    per_output = compute_anthropic_cost_usd(
        model=REFERENCE_MODEL, prompt_tokens=0, completion_tokens=_PER_MTOK
    )
    return per_input or 0.0, per_output or 0.0


def _reference_cost_usd(call: CallUsage) -> float:
    reference_in, reference_out = _reference_rates()
    return (
        call.prompt_tokens * reference_in + call.output_tokens * reference_out
    ) / _PER_MTOK


def _count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and value > 0 else 0
