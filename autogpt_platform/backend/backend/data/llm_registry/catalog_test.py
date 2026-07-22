"""Forever-guards for the canonical catalog file.

These tests ARE the review for catalog-only PRs (the /review bot fast-path
relies on them): if one fails, the catalog edit is wrong — fix the data,
never the test.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.blocks.llm import MODEL_METADATA, LLMModel
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


# NOTE: there is deliberately no "matrix fully specified" guard. Cells ship
# empty and get claimed one at a time — an unset cell means env vars keep
# that (mode, tier), which is the intended rollout-safe default. Cells that
# DO exist are governed by the reference and spelling tests above/below.


_SNAPSHOT_PATH = Path(__file__).parent / "pre_catalog_costs_snapshot.json"


def test_billing_matches_pre_catalog_snapshot():
    """Cutover-parity proof: the catalog-derived cost dicts reproduce the
    exact prices billed by the hand-maintained literals they replaced
    (snapshot captured 2026-07-20). New models add entries freely; changing
    a pre-cutover model's price is a deliberate two-line diff (catalog +
    snapshot) so review sees old→new explicitly. Deletable once the
    cutover has soaked a release.
    """
    snapshot = json.loads(_SNAPSHOT_PATH.read_text())
    for slug, credits in snapshot["model_cost"].items():
        assert MODEL_COST[LLMModel(slug)] == credits, slug
    for slug, rate in snapshot["token_cost"].items():
        assert TOKEN_COST[LLMModel(slug)].model_dump() == rate, slug
    # Absence parity: the cutover itself must not silently move a model
    # between flat-rate and token billing.
    pre_cutover = set(snapshot["model_cost"])
    token_billed = {m.value for m in TOKEN_COST}
    assert token_billed & pre_cutover == set(snapshot["token_cost"])


def test_metadata_matches_pre_catalog_snapshot():
    """Same parity proof for the block-facing metadata projection.

    DISCLOSED DELTA: 7 display cells in the snapshot were updated to the
    catalog's names — the cutover intentionally renames them:
    provider_name "V0" → "v0 by Vercel" and creator_name "V0" →
    "v0 by Vercel" (v0-1.0-md, v0-1.5-md, v0-1.5-lg), plus provider_name
    "AI/ML" → "AI/ML API" (meta-llama/Llama-3.3-70B-Instruct-Turbo).
    Every other cell is the deleted literal, byte-for-byte.
    """
    snapshot = json.loads(_SNAPSHOT_PATH.read_text())
    for slug, fields in snapshot["model_metadata"].items():
        assert MODEL_METADATA[LLMModel(slug)]._asdict() == fields, slug


def test_exactly_one_enabled_recommended_model():
    """DEFAULT_LLM_MODEL derives from is_recommended in catalog order — a
    second recommended entry would silently shift the platform default."""
    recommended = [m.slug for m in CATALOG.models if m.is_recommended and m.is_enabled]
    assert len(recommended) == 1, recommended


def test_kimi_k3_bills_at_authored_rates():
    """The flagship catalog-native model's billing projections — flat tier
    and per-1M token rates — must match its authored catalog entry."""
    k3 = LLMModel("moonshotai/kimi-k3")
    assert MODEL_COST[k3] == 9
    assert TOKEN_COST[k3].model_dump() == {
        "input": 450.0,
        "output": 2250.0,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }


def test_provider_usd_prices_are_all_or_nothing():
    """A half-authored provider USD price must refuse to construct — it
    would silently underprice against the transport family default."""
    import pytest

    from backend.data.llm_registry.catalog_model import CatalogModelCost

    with pytest.raises(ValueError, match="must be set together"):
        CatalogModelCost(run_credits=1, provider_input_usd_per_1m=3.0)


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
