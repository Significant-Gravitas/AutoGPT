"""Live spend for a running turn, priced call by call.

The CLI reports what a turn cost only once it ends. Landing a task before
the money runs out needs a running figure, so each model call the CLI
streams is priced as it arrives, with the same rate cards the post-turn
estimate uses.

Two sources, keyed by the call's id with each count kept at its highest
value, since a call is reported several times:

* ``AssistantMessage`` (once per content block, for the main agent and for
  foreground sub-agents alike) has the id, the model and the input and cache
  counts, but its ``output_tokens`` is only the placeholder from the start
  of the response.
* The raw stream (``StreamEvent``, with partial messages on, the default)
  carries the real output count in each response's ``message_delta``, which
  names no call: it belongs to the last ``message_start`` on the same
  stream, the main agent's or a sub-agent's (by its parent tool call).

With partial messages off, output reads as the placeholder and the figure
runs low; the CLI backstop still bounds the turn. Compaction calls aren't
reported either. Billing still comes from the post-turn reconcile; this
only decides when to land.
"""

import re
from functools import cache
from typing import Any

from claude_agent_sdk import AssistantMessage, StreamEvent
from pydantic import BaseModel

from backend.copilot.anthropic_rate_card import compute_anthropic_cost_usd
from backend.copilot.moonshot import rate_card_usd
from backend.data.llm_registry.catalog import get_catalog

# The CLI's price table only knows Anthropic models and prices anything else
# like Sonnet. Also our price for a model no rate card names: overestimating
# lands a task early, underestimating overspends.
REFERENCE_MODEL = "claude-sonnet-4-5"

_PER_MTOK = 1_000_000

# A dated snapshot (``claude-sonnet-4-5-20250929``) costs what its alias does.
_SNAPSHOT_DATE = re.compile(r"-\d{8}$")


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
        self._spent = 0.0
        self._last_main_id: str | None = None
        # The call each stream is generating, by parent tool call (None for
        # the main agent): a ``message_delta`` doesn't say which it belongs to.
        self._streaming: dict[str | None, str] = {}

    def observe(self, message: AssistantMessage) -> None:
        if not message.message_id or not message.usage:
            return
        self._record(
            message.message_id,
            message.usage,
            model=message.model,
            main=message.parent_tool_use_id is None,
        )

    def observe_stream(self, event: StreamEvent) -> None:
        data = event.event
        stream = event.parent_tool_use_id
        if data.get("type") == "message_start":
            message = data.get("message")
            if not isinstance(message, dict) or not message.get("id"):
                return
            self._streaming[stream] = message["id"]
            self._record(
                message["id"],
                message.get("usage") or {},
                model=message.get("model"),
                main=stream is None,
            )
        elif data.get("type") == "message_delta":
            call_id = self._streaming.get(stream)
            usage = data.get("usage")
            if call_id is not None and isinstance(usage, dict):
                self._record(call_id, usage)

    def _record(
        self,
        call_id: str,
        usage: dict[str, Any],
        *,
        model: str | None = None,
        main: bool = True,
    ) -> None:
        call = self._calls.get(call_id)
        if call is None:
            call = CallUsage(model=model or None, main=main)
            self._calls[call_id] = call
        call.merge(usage)
        priced = price_call_usd(call)
        self._spent += priced - self._priced.get(call_id, 0.0)
        self._priced[call_id] = priced
        if call.main:
            self._last_main_id = call_id

    @property
    def spent_usd(self) -> float:
        return self._spent

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
        model=_rate_card_model(call.model),
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
            model=_rate_card_model(model), prompt_tokens=0, completion_tokens=0
        )
        is not None
    )


def _rate_card_model(model: str | None) -> str:
    """The Anthropic rate card's name for *model*.

    The CLI reports each call's model as its route names it: OpenRouter's
    ``anthropic/claude-sonnet-5-5``, a dotted ``claude-sonnet-4.5`` or the
    API's dated snapshot. The card keys the bare alias, and prices a model it
    can't find as Opus 4.1, several times Sonnet. Other vendors' slugs pass
    through to their own rates.
    """
    if not model:
        return REFERENCE_MODEL
    vendor, slash, name = model.partition("/")
    if not slash:
        name = model
    elif vendor.lower() != "anthropic":
        return model
    if not name.lower().startswith("claude-"):
        return model
    return _SNAPSHOT_DATE.sub("", name.lower().replace(".", "-"))


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
