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

# open_router models bill via COST_USD against the response's x-total-cost,
# never from TOKEN_COST (block_cost_config._open_router_llm_cost). Their
# per-1M rates are display only, and OpenRouter reprices continuously, so
# pinning those values to the cutover snapshot pins a number that is
# expected to drift and that no user is ever charged. Their *presence* in
# the snapshot stays pinned by the absence-parity check below; only the
# values are exempt. Every genuinely billed provider keeps full parity.
_DISPLAY_ONLY_TOKEN_COST_SLUGS = frozenset(
    m.slug for m in CATALOG.models if m.provider == "open_router"
)


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
        if slug in _DISPLAY_ONLY_TOKEN_COST_SLUGS:
            continue
        assert TOKEN_COST[LLMModel(slug)].model_dump() == rate, slug
    # Absence parity: the cutover itself must not silently move a model
    # between flat-rate and token billing. Presence is still pinned for the
    # display-only slugs above — only their values are allowed to move.
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
    and per-1M token rates — must match its authored catalog entry.

    Re-checked live against OpenRouter on 2026-09-25: still $3.00/$15.00
    per Mtok, no drift. Cache read is newly authored at $0.30/1M (was
    previously unpublished/unset)."""
    k3 = LLMModel("moonshotai/kimi-k3")
    assert MODEL_COST[k3] == 9
    assert TOKEN_COST[k3].model_dump() == {
        "input": 450.0,
        "output": 2250.0,
        "cache_read": 45.0,
        "cache_creation": 0.0,
    }


def test_deepseek_chat_display_rate_tracks_openrouter():
    """SECRT-2701 regression pin: deepseek-chat's shown price had drifted to
    ~3x understated ($0.14/$0.28 displayed against a live $0.32/$0.89) and
    nothing caught it. These figures are display only — open_router settles
    COST_USD against x-total-cost — but a wrong number shown before the user
    picks a model is still wrong. Re-derive with
    ``poetry run python scripts/check_openrouter_prices.py`` and move both
    sides together when OpenRouter reprices.
    """
    chat = LLMModel("deepseek/deepseek-chat")
    assert TOKEN_COST[chat].model_dump() == {
        "input": 48.0,  # $0.32/1M x 150 cr/$
        "output": 133.5,  # $0.89/1M x 150 cr/$
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }


def test_claude_sonnet_5_bills_at_authored_rates():
    """Sonnet 5 (sticker $3/$15; intro pricing ends 2026-08-31) — flat tier
    and per-1M projections must match the authored catalog entry."""
    s5 = LLMModel("claude-sonnet-5")
    assert MODEL_COST[s5] == 9
    assert TOKEN_COST[s5].model_dump() == {
        "input": 450.0,
        "output": 2250.0,
        "cache_read": 45.0,
        "cache_creation": 563.0,
    }
    assert MODEL_METADATA[s5].max_output_tokens == 128000


def test_claude_opus_5_bills_at_authored_rates():
    opus = LLMModel("claude-opus-5")
    assert MODEL_COST[opus] == 14
    assert TOKEN_COST[opus].model_dump() == {
        "input": 750.0,
        "output": 3750.0,
        "cache_read": 75.0,
        "cache_creation": 938.0,
    }
    assert MODEL_METADATA[opus].max_output_tokens == 128000


def test_claude_opus_5_5_bills_at_authored_rates():
    """Claude Opus 5.5 (Anthropic list price $4/$20 per 1M, undercutting
    Opus 5's $5/$25) — flat tier and per-1M projections must match the
    authored catalog entry."""
    opus55 = LLMModel("claude-opus-5-5")
    assert MODEL_COST[opus55] == 11
    assert TOKEN_COST[opus55].model_dump() == {
        "input": 600.0,
        "output": 3000.0,
        "cache_read": 30.0,
        "cache_creation": 750.0,
    }
    assert MODEL_METADATA[opus55].max_output_tokens == 128000
    opus55_entry = next(m for m in CATALOG.models if m.slug == "claude-opus-5-5")
    assert opus55_entry.price_tier == 3
    assert opus55_entry.context_window == 200000


def test_gpt6_astra_bills_at_authored_rates():
    """GPT-6 Astra (OpenAI list price $10/$50 per 1M) — flat tier and
    per-1M projections must match the authored catalog entry."""
    astra = LLMModel("gpt-6-astra")
    assert MODEL_COST[astra] == 20
    assert TOKEN_COST[astra].model_dump() == {
        "input": 1500.0,
        "output": 7500.0,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[astra].max_output_tokens == 128000
    astra_entry = next(m for m in CATALOG.models if m.slug == "gpt-6-astra")
    assert astra_entry.price_tier == 3
    assert astra_entry.context_window == 1050000


def test_gpt6_sol_bills_at_authored_rates():
    """GPT-6 Sol (OpenAI list price $2/$10 per 1M, the cost-efficient
    high-end tier below Astra) — flat tier and per-1M projections must
    match the authored catalog entry."""
    sol = LLMModel("gpt-6-sol")
    assert MODEL_COST[sol] == 4
    assert TOKEN_COST[sol].model_dump() == {
        "input": 300.0,
        "output": 1500.0,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[sol].max_output_tokens == 128000
    sol_entry = next(m for m in CATALOG.models if m.slug == "gpt-6-sol")
    assert sol_entry.price_tier == 2
    assert sol_entry.context_window == 1050000


def test_gpt6_luna_bills_at_authored_rates():
    """GPT-6 Luna (OpenAI list price $0.10/$0.50 per 1M, the fast/cheapest
    GPT-6 tier) — flat tier and per-1M projections must match the
    authored catalog entry."""
    luna = LLMModel("gpt-6-luna")
    assert MODEL_COST[luna] == 1
    assert TOKEN_COST[luna].model_dump() == {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[luna].max_output_tokens == 128000
    luna_entry = next(m for m in CATALOG.models if m.slug == "gpt-6-luna")
    assert luna_entry.price_tier == 1
    assert luna_entry.context_window == 1050000


def test_claude_fable_5_1_bills_at_authored_rates():
    """Claude Fable 5.1 (Anthropic list price $10/$50 per 1M, cache reads cut
    75% to $0.25/1M) — flat tier and per-1M projections must match the
    authored catalog entry."""
    fable = LLMModel("claude-fable-5-1")
    assert MODEL_COST[fable] == 20
    assert TOKEN_COST[fable].model_dump() == {
        "input": 1500.0,
        "output": 7500.0,
        "cache_read": 37.5,
        "cache_creation": 1875.0,
    }
    assert MODEL_METADATA[fable].max_output_tokens == 128000
    fable_entry = next(m for m in CATALOG.models if m.slug == "claude-fable-5-1")
    assert fable_entry.price_tier == 3
    assert fable_entry.context_window == 200000


def test_gemini_3_8_flash_bills_at_authored_rates():
    """Gemini 3.8 Flash (OpenRouter, Google intro list price $0.75/$3.75
    per 1M through 2026-12-31) — flat tier and per-1M projections must
    match the authored catalog entry."""
    flash = LLMModel("google/gemini-3.8-flash")
    assert MODEL_COST[flash] == 3
    assert TOKEN_COST[flash].model_dump() == {
        "input": 112.5,
        "output": 562.5,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[flash].max_output_tokens == 65536
    flash_entry = next(m for m in CATALOG.models if m.slug == "google/gemini-3.8-flash")
    assert flash_entry.price_tier == 1
    assert flash_entry.context_window == 1048576


def test_muse_spark_1_3_bills_at_authored_rates():
    """Muse Spark 1.3 (OpenRouter, Meta list price $1.25/$4.25 per 1M) —
    flat tier and per-1M projections must match the authored catalog
    entry."""
    muse_spark = LLMModel("meta/muse-spark-1.3")
    assert MODEL_COST[muse_spark] == 3
    assert TOKEN_COST[muse_spark].model_dump() == {
        "input": 187.5,
        "output": 637.5,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[muse_spark].max_output_tokens == 1000000
    muse_spark_entry = next(
        m for m in CATALOG.models if m.slug == "meta/muse-spark-1.3"
    )
    assert muse_spark_entry.price_tier == 1
    assert muse_spark_entry.context_window == 1048576
    assert muse_spark_entry.supports_tools is True
    assert muse_spark_entry.supports_json_output is True
    assert muse_spark_entry.supports_reasoning is True


def test_muse_spark_1_3_contributor_bills_at_authored_rates():
    """Muse Spark 1.3 Contributor (OpenRouter, Meta list price
    $0.10/$0.20 per 1M, $0.002/1M cached input) — the discounted
    data-sharing tier — flat tier and per-1M projections must match the
    authored catalog entry."""
    contributor = LLMModel("meta/muse-spark-1.3-contributor")
    assert MODEL_COST[contributor] == 1
    assert TOKEN_COST[contributor].model_dump() == {
        "input": 15.0,
        "output": 30.0,
        "cache_read": 0.3,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[contributor].max_output_tokens == 1000000
    contributor_entry = next(
        m for m in CATALOG.models if m.slug == "meta/muse-spark-1.3-contributor"
    )
    assert contributor_entry.price_tier == 1
    assert contributor_entry.context_window == 1048576
    assert contributor_entry.supports_tools is True
    assert contributor_entry.supports_json_output is True
    assert contributor_entry.supports_reasoning is True


def test_qwen3_8_max_0902_bills_at_authored_rates():
    """Qwen 3.8 Max (0902) (OpenRouter, Alibaba list price $2.00/$6.00 per
    1M) — flat tier and per-1M projections must match the authored catalog
    entry."""
    qwen_max = LLMModel("qwen/qwen3.8-max-0902")
    assert MODEL_COST[qwen_max] == 5
    assert TOKEN_COST[qwen_max].model_dump() == {
        "input": 300.0,
        "output": 900.0,
        "cache_read": 0.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[qwen_max].max_output_tokens == 131072
    qwen_max_entry = next(
        m for m in CATALOG.models if m.slug == "qwen/qwen3.8-max-0902"
    )
    assert qwen_max_entry.price_tier == 2
    assert qwen_max_entry.context_window == 262144


def test_qwen3_8_flash_bills_at_authored_rates():
    """Qwen 3.8 Flash (OpenRouter, live list price $0.15/$0.47 per 1M,
    $0.016/1M cached input, $0.20/1M cache write) — flat tier and per-1M
    projections must match the authored catalog entry."""
    qwen_flash = LLMModel("qwen/qwen3.8-flash")
    assert MODEL_COST[qwen_flash] == 1
    assert TOKEN_COST[qwen_flash].model_dump() == {
        "input": 22.5,
        "output": 70.5,
        "cache_read": 2.4,
        "cache_creation": 30.0,
    }
    assert MODEL_METADATA[qwen_flash].max_output_tokens == 131072
    qwen_flash_entry = next(m for m in CATALOG.models if m.slug == "qwen/qwen3.8-flash")
    assert qwen_flash_entry.price_tier == 1
    assert qwen_flash_entry.context_window == 1000000
    assert qwen_flash_entry.supports_tools is True
    assert qwen_flash_entry.supports_json_output is True
    assert qwen_flash_entry.supports_reasoning is True


def test_deepseek_v4_1_flash_bills_at_authored_rates():
    """DeepSeek V4.1 Flash (OpenRouter live rate $0.14/$0.42 per 1M,
    $0.0042/1M cached input as of 2026-09-25 — this route reprices
    continuously by design) — flat tier and per-1M projections must
    match the authored catalog entry."""
    flash = LLMModel("deepseek/deepseek-v4.1-flash")
    assert MODEL_COST[flash] == 1
    assert TOKEN_COST[flash].model_dump() == {
        "input": 21.0,
        "output": 63.0,
        "cache_read": 0.63,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[flash].max_output_tokens == 384000
    flash_entry = next(
        m for m in CATALOG.models if m.slug == "deepseek/deepseek-v4.1-flash"
    )
    assert flash_entry.price_tier == 1
    assert flash_entry.context_window == 1048576


def test_fugu_ultra_v2_bills_at_authored_rates():
    """Sakana Fugu Ultra v2 (OpenRouter, Sakana list price $5.00/$30.00 per
    1M, $0.50/1M cached input) — flat tier and per-1M projections must
    match the authored catalog entry."""
    fugu = LLMModel("sakana/fugu-ultra-v2")
    assert MODEL_COST[fugu] == 1
    assert TOKEN_COST[fugu].model_dump() == {
        "input": 750.0,
        "output": 4500.0,
        "cache_read": 75.0,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[fugu].max_output_tokens == 1048576
    fugu_entry = next(m for m in CATALOG.models if m.slug == "sakana/fugu-ultra-v2")
    assert fugu_entry.price_tier == 3
    assert fugu_entry.context_window == 1048576
    assert fugu_entry.supports_tools is True
    assert fugu_entry.supports_json_output is True
    assert fugu_entry.supports_reasoning is True


def test_mercury_2_5_bills_at_authored_rates():
    """Inception Mercury 2.5 (OpenRouter, Inception list price $0.04/$0.15
    per 1M, $0.004/1M cached input) — flat tier and per-1M projections must
    match the authored catalog entry."""
    mercury = LLMModel("inception/mercury-2.5")
    assert MODEL_COST[mercury] == 1
    assert TOKEN_COST[mercury].model_dump() == {
        "input": 6.0,
        "output": 22.5,
        "cache_read": 0.6,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[mercury].max_output_tokens == 65536
    mercury_entry = next(m for m in CATALOG.models if m.slug == "inception/mercury-2.5")
    assert mercury_entry.price_tier == 1
    assert mercury_entry.context_window == 260000
    assert mercury_entry.supports_tools is True
    assert mercury_entry.supports_json_output is True
    assert mercury_entry.supports_reasoning is True
    assert mercury_entry.supports_parallel_tool_calls is True


def test_hy4_preview_bills_at_authored_rates():
    """Tencent Hy4 Preview (OpenRouter, live list price $0.834/$2.501 per
    1M, $0.042/1M cached input) — flat tier and per-1M projections must
    match the authored catalog entry."""
    hy4 = LLMModel("tencent/hy4-preview")
    assert MODEL_COST[hy4] == 2
    assert TOKEN_COST[hy4].model_dump() == {
        "input": 125.1,
        "output": 375.15,
        "cache_read": 6.3,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[hy4].max_output_tokens == 64000
    hy4_entry = next(m for m in CATALOG.models if m.slug == "tencent/hy4-preview")
    assert hy4_entry.price_tier == 2
    assert hy4_entry.context_window == 1048576
    assert hy4_entry.supports_tools is True
    assert hy4_entry.supports_reasoning is True


def test_pareto_bills_at_authored_rates():
    """Unbiased Pareto (OpenRouter, list price $2.50/$7.50 per 1M, $0.25/1M
    cached input) — flat tier and per-1M projections must match the
    authored catalog entry."""
    pareto = LLMModel("unbiased/pareto")
    assert MODEL_COST[pareto] == 1
    assert TOKEN_COST[pareto].model_dump() == {
        "input": 375.0,
        "output": 1125.0,
        "cache_read": 37.5,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[pareto].max_output_tokens == 131072
    pareto_entry = next(m for m in CATALOG.models if m.slug == "unbiased/pareto")
    assert pareto_entry.price_tier == 2
    assert pareto_entry.context_window == 262144
    assert pareto_entry.supports_tools is True


def test_ling_3_0_flash_vl_bills_at_authored_rates():
    """InclusionAI Ling 3.0 Flash VL (OpenRouter live rate $0.021/$0.0616
    per 1M, $0.0042/1M cached input as of 2026-09-25 — dropped from
    $0.06/$0.18) — flat tier and per-1M projections must match the
    authored catalog entry."""
    ling = LLMModel("inclusionai/ling-3.0-flash-vl")
    assert MODEL_COST[ling] == 1
    assert TOKEN_COST[ling].model_dump() == {
        "input": 3.15,
        "output": 9.24,
        "cache_read": 0.63,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[ling].max_output_tokens == 32768
    ling_entry = next(
        m for m in CATALOG.models if m.slug == "inclusionai/ling-3.0-flash-vl"
    )
    assert ling_entry.price_tier == 1
    assert ling_entry.context_window == 131072
    assert ling_entry.supports_tools is True
    assert ling_entry.supports_json_output is True
    assert ling_entry.supports_reasoning is True


def test_gemma_4_31b_it_bills_at_authored_rates():
    """Google Gemma 4 31B (OpenRouter live rate $0.09/$0.34 per 1M,
    $0.05/1M cached input as of 2026-09-23) — flat tier and per-1M
    projections must match the authored catalog entry."""
    gemma = LLMModel("google/gemma-4-31b-it")
    assert MODEL_COST[gemma] == 1
    assert TOKEN_COST[gemma].model_dump() == {
        "input": 13.5,
        "output": 51.0,
        "cache_read": 7.5,
        "cache_creation": 0.0,
    }
    assert MODEL_METADATA[gemma].max_output_tokens == 16384
    gemma_entry = next(m for m in CATALOG.models if m.slug == "google/gemma-4-31b-it")
    assert gemma_entry.price_tier == 1
    assert gemma_entry.context_window == 262144


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
