"""Forever-guards for the canonical catalog file.

These tests ARE the review for catalog-only PRs (the /review bot fast-path
relies on them): if one fails, the catalog edit is wrong — fix the data,
never the test.
"""

from __future__ import annotations

from backend.blocks.llm import LlmModel
from backend.data.block_cost_config import MODEL_COST, TOKEN_COST
from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CATALOG_SCHEMA_VERSION

CATALOG = get_catalog()


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


def _resolve_cell(by_slug: dict, value: str):
    """Slug-tolerant cell resolution mirroring the router's gate: exact,
    vendor-stripped, and dots-to-dashes forms (cells carry transport
    spellings, not canonical slugs — see the spelling-convention test)."""
    candidates = {value, value.split("/", 1)[-1]}
    candidates |= {c.replace(".", "-") for c in set(candidates)}
    for c in candidates:
        if c in by_slug:
            return by_slug[c]
    return None


def test_routing_cells_reference_enabled_models():
    by_slug = {m.slug: m for m in CATALOG.models}
    for surface, modes in CATALOG.routing.items():
        for mode, tiers in modes.items():
            for tier, slug in tiers.items():
                cell = f"routing[{surface}][{mode}][{tier}]"
                model = _resolve_cell(by_slug, slug)
                assert model is not None, f"{cell}: unknown model {slug}"
                assert model.is_enabled, f"{cell}: model {slug} is disabled"


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


def test_routing_cells_use_transport_ready_spellings():
    """Routing cells are sent to providers (nearly) verbatim, so they must use
    the spelling the transport expects — NOT the catalog's canonical slug.

    Convention: Anthropic cells use the vendor-prefixed DOT form
    (``anthropic/claude-sonnet-4.6``) — the form OpenRouter serves and the
    direct-Anthropic normalizer accepts. Bare ``claude-*`` 404s on OpenRouter;
    dash-form ``anthropic/claude-*-4-6`` exists on no transport. Cells still
    gate against the catalog via the resolver's slug-tolerant lookup.
    """
    slugs = {m.slug for m in CATALOG.models}

    def tolerant_match(value: str) -> bool:
        candidates = {value, value.split("/", 1)[-1]}
        candidates |= {c.replace(".", "-") for c in set(candidates)}
        return any(c in slugs for c in candidates)

    for surface, modes in CATALOG.routing.items():
        for mode, tiers in modes.items():
            for tier, cell in tiers.items():
                where = f"routing[{surface}][{mode}][{tier}] = {cell!r}"
                assert tolerant_match(cell), f"{where} matches no catalog model"
                assert not cell.startswith("claude-"), (
                    f"{where}: bare claude-* cells 404 on OpenRouter — use "
                    "the vendor-prefixed dot form (anthropic/claude-…4.6)"
                )
                if cell.startswith("anthropic/"):
                    tail = cell.split("/", 1)[1]
                    assert "." in tail, (
                        f"{where}: dash-form anthropic/ cells exist on no "
                        "transport — use the dot form (anthropic/claude-…4.6)"
                    )
