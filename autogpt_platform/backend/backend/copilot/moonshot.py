"""Moonshot-specific pricing and cache-control helpers.

Moonshot's Kimi K2.x family is routed through OpenRouter's Anthropic-compat
shim — it speaks Anthropic's API shape but its pricing and cache behaviour
diverge from Anthropic in ways the Claude Agent SDK CLI and our baseline
cache-control gating don't handle on their own:

* **Rate card** — NOT the canonical cost source.  The authoritative number
  for every OpenRouter-routed turn is the reconcile task
  (:mod:`openrouter_cost`), which reads ``total_cost`` directly from
  ``/api/v1/generation`` post-turn.  This module exists purely so the
  CLI's in-turn ``ResultMessage.total_cost_usd`` (which silently bills
  Moonshot at Sonnet rates, ~5x the real Moonshot price because the CLI
  pricing table only knows Anthropic) isn't left wildly wrong before the
  reconcile fires AND so the reconcile's lookup-fail fallback records a
  plausible Moonshot estimate rather than a Sonnet-rate overcharge.
  Signal authority: reconcile >> this module's rate card >> CLI.

* **Cache-control** — Anthropic and Moonshot both accept the
  ``cache_control: {type: ephemeral}`` breakpoint on message blocks, but
  our baseline path currently gates cache markers on an
  ``anthropic/`` / ``claude`` name match because non-Anthropic providers
  (OpenAI, Grok, Gemini) 400 on the unknown field.  Moonshot's
  Anthropic-compat endpoint silently accepts and honours the marker —
  empirically boosts cache hit rate on continuation turns — but was
  caught in the non-Anthropic branch of the original gate.
  :func:`moonshot_supports_cache_control` lets callers widen the gate
  to include Moonshot without weakening the ``false`` answer for
  OpenAI et al.  (The predicate is intentionally narrow — Moonshot-only
  — so callers combine it with an explicit Anthropic check at the call
  site; see ``baseline/service.py::_supports_prompt_cache_markers``.)

Detection is prefix-based (``moonshotai/``).  Moonshot routes every Kimi
SKU through the same Anthropic-compat surface, so a new slug transparently
inherits both the rate card and the cache-control gate without editing
this file.  SKUs priced off the shared K2.x default (Kimi K3 broke the
uniform pricing at 5x) carry per-slug entries in
:data:`_RATE_OVERRIDES_USD_PER_MTOK`.
"""

from __future__ import annotations

from backend.data.llm_registry.catalog import get_catalog

# Every Kimi K2.x SKU shares these rates — $0.60/$2.80 per million
# (input/output) via OpenRouter as of April 2026.
# Cache-read / cache-write discounts are NOT applied here:
# OpenRouter currently exposes only a single input price per Moonshot
# endpoint; the real billed amount (with cache savings) lands via the
# reconcile path.  Keep in sync with https://platform.moonshot.ai/docs/pricing.
_DEFAULT_MOONSHOT_RATE_USD_PER_MTOK: tuple[float, float] = (0.60, 2.80)


# Vendor prefix — matches any OpenRouter slug Moonshot ships.  Keep as a
# module constant so the prefix check stays in exactly one place.
_MOONSHOT_PREFIX = "moonshotai/"


def _overrides_from_catalog() -> dict[str, tuple[float, float]]:
    """Per-slug USD overrides for SKUs priced off the shared K2.x default.

    Authored in the catalog (``provider_*_usd_per_1m`` on the model's cost
    entry — e.g. Kimi K3's $3/$15); slugs without them fall back to
    :data:`_DEFAULT_MOONSHOT_RATE_USD_PER_MTOK`.
    """
    return {
        m.slug: (m.cost.provider_input_usd_per_1m, m.cost.provider_output_usd_per_1m)
        for m in get_catalog().models
        if m.slug.startswith(_MOONSHOT_PREFIX)
        and m.cost is not None
        and m.cost.provider_input_usd_per_1m is not None
        and m.cost.provider_output_usd_per_1m is not None
    }


# Import-time snapshot is safe: the catalog is immutable within a process
# (the file only changes at deploy), same lifecycle as every projection.
_RATE_OVERRIDES_USD_PER_MTOK: dict[str, tuple[float, float]] = _overrides_from_catalog()


def is_moonshot_model(model: str | None) -> bool:
    """True when *model* is a Moonshot OpenRouter slug.

    Prefix match against ``moonshotai/`` covers every Kimi SKU under that
    namespace — present and future — with no per-SKU roster to maintain.
    Used by both pricing and cache-control gating.
    """
    return isinstance(model, str) and model.startswith(_MOONSHOT_PREFIX)


def rate_card_usd(model: str | None) -> tuple[float, float] | None:
    """Return (input, output) $/Mtok for *model* or None if non-Moonshot.

    Looks up a per-slug override first, falling back to the shared
    default for anything under ``moonshotai/``.  Returns None for
    non-Moonshot slugs (including ``None``) so callers can skip the
    override without a preflight guard.
    """
    if not is_moonshot_model(model):
        return None
    # ``is_moonshot_model`` narrowed ``model`` to str; dict.get is
    # type-safe here despite the wider param annotation above.
    assert model is not None
    return _RATE_OVERRIDES_USD_PER_MTOK.get(model, _DEFAULT_MOONSHOT_RATE_USD_PER_MTOK)


def override_cost_usd(
    *,
    model: str | None,
    sdk_reported_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float:
    """Recompute SDK turn cost from the Moonshot rate card.

    Not the canonical cost source — the OpenRouter ``/generation``
    reconcile (:mod:`openrouter_cost`) lands the authoritative billed
    amount post-turn.  This helper exists only to improve the CLI's
    in-turn ``ResultMessage.total_cost_usd``:

    1. So the ``cost_usd`` the client sees before the reconcile completes
       isn't wildly wrong (the CLI would otherwise ship a Sonnet-rate
       estimate, ~5x the real Moonshot bill).
    2. So the reconcile's own lookup-fail fallback records a plausible
       Moonshot estimate rather than the CLI's Sonnet number.

    For Moonshot slugs we compute cost from the reported token counts;
    for anything else (including Anthropic) we return the SDK number
    unchanged — Anthropic slugs are priced accurately by the CLI.

    Cache read / creation tokens are folded into ``prompt_tokens`` at
    the full input rate because Moonshot's rate card doesn't distinguish
    them at the OpenRouter surface; the reconcile has the authoritative
    discount accounting for turns where Moonshot's cache engaged.
    """
    if model is None:
        return sdk_reported_usd
    rates = rate_card_usd(model)
    if rates is None:
        return sdk_reported_usd
    input_rate, output_rate = rates
    total_prompt = prompt_tokens + cache_read_tokens + cache_creation_tokens
    return (total_prompt * input_rate + completion_tokens * output_rate) / 1_000_000


def moonshot_supports_cache_control(model: str | None) -> bool:
    """True when a Moonshot *model* accepts Anthropic-style ``cache_control``.

    Narrow, Moonshot-specific predicate — callers that need the full
    "does this route accept cache markers" answer combine this with an
    Anthropic check (see ``baseline/service.py::_supports_prompt_cache_markers``).
    Named ``moonshot_*`` deliberately so the call site can't mistake it
    for a universal predicate that answers correctly for Anthropic
    (which also supports cache_control — this function would return
    False for Anthropic slugs).

    Moonshot's Anthropic-compat endpoint honours the marker.  Without
    it Moonshot falls back to its own automatic prefix caching, which
    drifts more readily between turns (internal testing saw 0/4 cache
    hits across two continuation sessions).  With explicit
    ``cache_control`` the upstream cache hit rate rises to the same
    ballpark as Anthropic's ~60-95% on continuations.
    """
    return is_moonshot_model(model)
