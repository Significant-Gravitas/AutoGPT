"""Guards for the background-inference price card.

The copilot model defaults (the standard, advanced, title and gate models)
and Graphiti's memory LLM are what runs when nothing overrides them. Each
must resolve to a catalog price, or a call priced from the card logs its
tokens and charges nothing.
"""

from __future__ import annotations

import logging

import pytest

from backend.copilot import price_card
from backend.copilot.config import ChatConfig
from backend.copilot.dream.routing import batch_discount
from backend.copilot.graphiti.config import GraphitiConfig
from backend.copilot.price_card import PriceCard, compute_cost_usd, price_for
from backend.data.llm_registry import registry
from backend.data.llm_registry.catalog_model import CatalogModel, CatalogModelCost

# ChatConfig ``*_model`` fields are picked up by these role words in their
# names, so a new tier or gate model is guarded without editing this file.
_PRICED_ROLES = ("standard", "advanced", "title", "gate")
# Role-named fields that are not LLMs from the catalog.
_NOT_CATALOG_LLMS = {"gate_jev_model": "Typesafe's Jev classifier"}


def _copilot_default_models() -> dict[str, str]:
    return {
        name: field.default
        for name, field in ChatConfig.model_fields.items()
        if name.endswith("_model")
        and set(name.split("_")) & set(_PRICED_ROLES)
        and name not in _NOT_CATALOG_LLMS
    }


def _background_default_models() -> dict[str, str]:
    graphiti_llm = GraphitiConfig.model_fields["llm_model"].default
    return {**_copilot_default_models(), "graphiti.llm_model": graphiti_llm}


@pytest.fixture(autouse=True)
def fresh_unpriced_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees the first-report-per-process error, whatever ran before."""
    monkeypatch.setattr(price_card, "_reported_unpriced", set())


def test_every_role_has_a_default_model():
    found = _copilot_default_models()
    for role in _PRICED_ROLES:
        assert any(role in name.split("_") for name in found), role
    assert all(isinstance(slug, str) and slug for slug in found.values()), found


@pytest.mark.parametrize("field,slug", sorted(_background_default_models().items()))
def test_background_default_model_is_priced(field: str, slug: str):
    assert price_for(slug) is not None, f"{field}={slug!r} has no catalog price"


@pytest.mark.parametrize(
    "spelling,catalog_slug",
    [
        ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
        ("claude-opus-5.5", "claude-opus-5-5"),
        ("anthropic/claude-opus-5.5", "claude-opus-5-5"),
        ("anthropic/claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"),
        ("openai/gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"),
        ("Anthropic/Claude-Sonnet-5", "claude-sonnet-5"),
    ],
)
def test_transport_spellings_resolve_to_the_catalog_entry(spelling, catalog_slug):
    card = price_for(spelling)
    assert card is not None
    assert card == price_for(catalog_slug)


def test_resolution_does_not_need_the_in_process_registry(monkeypatch):
    """The scheduler and the batch executor never run ``load_catalog()``,
    so their registry stays empty; pricing must not notice."""
    monkeypatch.setattr(registry, "_dynamic_models", {})
    monkeypatch.setattr(registry, "_date_stripped_models", {})
    monkeypatch.setattr(registry, "_loaded", False)
    price_card._catalog_index.cache_clear()
    assert not registry.has_models()

    for slug in _background_default_models().values():
        assert price_for(slug) is not None, slug


def test_sonnet_5_card_carries_the_authored_list_prices():
    assert price_for("anthropic/claude-sonnet-5") == PriceCard(
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cache_read_usd_per_mtok=0.30,
        cache_creation_usd_per_mtok=3.75,
    )


def test_each_bucket_is_priced_at_its_own_rate():
    card = price_for("claude-sonnet-5")
    assert card is not None
    one_million = 1_000_000
    assert compute_cost_usd(
        price=card, input_tokens=one_million, output_tokens=0
    ) == pytest.approx(3.0)
    assert compute_cost_usd(
        price=card,
        input_tokens=one_million,
        output_tokens=one_million,
        cache_read_tokens=one_million,
        cache_creation_tokens=one_million,
    ) == pytest.approx(3.0 + 15.0 + 0.30 + 3.75)


def test_batch_discount_halves_the_list_cost():
    card = price_for("claude-opus-5-5")
    assert card is not None
    tokens = dict(
        input_tokens=120_000,
        output_tokens=8_000,
        cache_read_tokens=40_000,
        cache_creation_tokens=10_000,
    )
    sync_cost = compute_cost_usd(
        price=card, discount=batch_discount("sync_baseline"), **tokens
    )
    batch_cost = compute_cost_usd(
        price=card, discount=batch_discount("anthropic_batch"), **tokens
    )
    # $4 in, $20 out, $0.20 cache read, $5 cache write per Mtok.
    assert sync_cost == pytest.approx(0.48 + 0.16 + 0.008 + 0.05)
    assert batch_cost == pytest.approx(sync_cost / 2)


@pytest.mark.parametrize("discount", [-0.1, 1.5])
def test_discount_outside_zero_to_one_is_refused(discount):
    card = price_for("claude-sonnet-5")
    assert card is not None
    with pytest.raises(ValueError, match="discount"):
        compute_cost_usd(price=card, input_tokens=1, output_tokens=1, discount=discount)


def test_unknown_model_is_unpriced_and_reported_once_at_error(caplog):
    with caplog.at_level(logging.WARNING, logger=price_card.__name__):
        assert price_for("vendor/no-such-model") is None
        assert price_for("vendor/no-such-model") is None

    levels = [r.levelno for r in caplog.records if "no-such-model" in r.message]
    assert levels == [logging.ERROR, logging.WARNING]


def test_catalog_entry_without_every_usd_price_is_unpriced(monkeypatch, caplog):
    """A missing cache price would have to be invented or read as free, so
    the card needs all four buckets."""
    partial = CatalogModel(
        slug="claude-partial-1",
        display_name="Partial",
        provider="anthropic",
        context_window=1000,
        cost=CatalogModelCost(
            provider_input_usd_per_1m=1.0, provider_output_usd_per_1m=5.0
        ),
    )
    monkeypatch.setattr(
        price_card, "_catalog_index", lambda: ({partial.slug: partial}, {})
    )
    with caplog.at_level(logging.ERROR, logger=price_card.__name__):
        assert price_for("claude-partial-1") is None
    assert "claude-partial-1" in caplog.text
