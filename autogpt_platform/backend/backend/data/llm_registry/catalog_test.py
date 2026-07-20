"""Forever-guards for the canonical catalog file.

These tests ARE the review for catalog-only PRs (the /review bot fast-path
relies on them): if one fails, the catalog edit is wrong — fix the data,
never the test.
"""

from __future__ import annotations

from backend.blocks.llm import LlmModel
from backend.data.block_cost_config import MODEL_COST, TOKEN_COST
from backend.data.llm_registry.catalog import CATALOG
from backend.data.llm_registry.catalog_model import CATALOG_SCHEMA_VERSION


def test_catalog_declares_current_schema_version():
    assert CATALOG.schema_version == CATALOG_SCHEMA_VERSION


def test_catalog_is_not_suspiciously_small():
    assert len(CATALOG.models) > 50, "catalog lost most of its models"


def test_model_slugs_are_unique():
    slugs = [m.slug for m in CATALOG.models]
    assert len(slugs) == len(set(slugs)), "duplicate model slugs"


def test_provider_and_creator_references_resolve():
    provider_names = {p.name for p in CATALOG.providers}
    creator_names = {c.name for c in CATALOG.creators}
    for m in CATALOG.models:
        assert m.provider in provider_names, f"{m.slug}: unknown provider {m.provider}"
        if m.creator:
            assert m.creator in creator_names, f"{m.slug}: unknown creator {m.creator}"


def test_fallback_references_resolve():
    slugs = {m.slug for m in CATALOG.models}
    for m in CATALOG.models:
        if m.fallback_model_slug:
            assert (
                m.fallback_model_slug in slugs
            ), f"{m.slug}: fallback {m.fallback_model_slug} is not in the catalog"
            assert m.fallback_model_slug != m.slug, f"{m.slug}: fallback is itself"


def test_routing_cells_reference_enabled_models():
    by_slug = {m.slug: m for m in CATALOG.models}
    for surface, modes in CATALOG.routing.items():
        for mode, tiers in modes.items():
            for tier, slug in tiers.items():
                cell = f"routing[{surface}][{mode}][{tier}]"
                assert slug in by_slug, f"{cell}: unknown model {slug}"
                assert by_slug[slug].is_enabled, f"{cell}: model {slug} is disabled"


def test_copilot_routing_matrix_is_fully_specified():
    copilot = CATALOG.routing.get("copilot", {})
    for mode in ("fast", "thinking"):
        for tier in ("standard", "advanced"):
            assert copilot.get(mode, {}).get(tier), (
                f"copilot routing cell ({mode}, {tier}) is unset — the file is "
                "the config layer between LD and env defaults; leaving a cell "
                "empty silently shifts control to env vars"
            )


def test_cost_drift_tripwire_run_credits():
    """Catalog costs must equal the live billing dicts.

    The catalog centralizes per-model pricing NOW, but billing still reads
    ``MODEL_COST``/``TOKEN_COST`` until Phase B3 flips the reader and deletes
    the dicts (this test dies with them). Until then any change to one side
    without the other fails here instead of silently diverging.
    """
    by_slug = {m.slug: m for m in CATALOG.models}
    for member, credits in MODEL_COST.items():
        model = by_slug.get(member.value)
        assert model is not None, f"{member.value} priced in MODEL_COST, not in catalog"
        assert model.cost is not None, f"{member.value}: catalog entry has no cost"
        assert model.cost.run_credits == credits, (
            f"{member.value}: catalog run_credits={model.cost.run_credits} "
            f"!= MODEL_COST={credits}"
        )


def test_cost_drift_tripwire_token_rates():
    """Per-1M token rates must equal TOKEN_COST (see run_credits tripwire)."""
    by_slug = {m.slug: m for m in CATALOG.models}
    for member, rate in TOKEN_COST.items():
        model = by_slug.get(member.value)
        assert (
            model is not None and model.cost is not None
        ), f"{member.value} priced in TOKEN_COST, missing catalog cost"
        assert model.cost.input_credits_per_1m == rate.input, member.value
        assert model.cost.output_credits_per_1m == rate.output, member.value
        assert (
            model.cost.cache_read_credits_per_1m or 0
        ) == rate.cache_read, member.value
        assert (
            model.cost.cache_creation_credits_per_1m or 0
        ) == rate.cache_creation, member.value


def test_cost_drift_tripwire_reverse_direction():
    """Every catalog cost corresponds to a real enum model still priced in
    the dicts — a model removed from code must not keep a ghost price here."""
    enum_slugs = {m.value for m in LlmModel}
    priced_slugs = {m.value for m in MODEL_COST}
    for model in CATALOG.models:
        if model.cost is None:
            continue
        assert (
            model.slug in enum_slugs
        ), f"{model.slug} has a cost but is not an LlmModel enum member"
        if model.cost.run_credits is not None:
            assert (
                model.slug in priced_slugs
            ), f"{model.slug} has run_credits but no MODEL_COST entry"
