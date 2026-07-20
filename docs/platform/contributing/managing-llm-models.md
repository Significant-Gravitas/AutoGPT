# Managing LLM Models

## Overview

The platform manages LLM models **catalog-as-code**: one canonical, schema-validated file is the source of truth for model definitions, per-model costs, and AutoPilot (copilot) routing. There is no admin UI and no model database — you change models by editing the catalog and opening a PR, git history is the audit log, and the normal deploy pipeline propagates the change to every environment.

The catalog lives at:

```
autogpt_platform/backend/backend/data/llm_registry/catalog.py
```

Its schema is defined in `catalog_model.py` (same directory), and `catalog_test.py` contains the integrity guards — the file must parse, slugs must be unique, every provider/creator/fallback/routing reference must resolve, and costs must stay within bounds. A catalog PR that passes these tests is structurally sound by construction, which is what makes bot-reviewed catalog changes safe.

## Catalog fields

Each `CatalogModel` entry:

| Field | Meaning |
| --- | --- |
| `slug` | Canonical model identifier (e.g. `claude-sonnet-4-6`, `openai/gpt-5.2`). Referenced by routing cells and fallbacks. |
| `display_name` | Human-readable name shown in UIs. |
| `provider` | Who serves the model (must match a `CatalogProvider.name`). Determines which credential/API key is used. |
| `creator` | Who trained the model (display metadata; must match a `CatalogCreator.name`). |
| `kind` | Model modality; currently `CHAT`. |
| `context_window` / `max_output_tokens` | Token limits. |
| `price_tier` | 1 (cheapest) to 3 (most expensive); used for display. |
| `is_enabled` | **The kill switch.** A disabled model is refused at serve time — even when LaunchDarkly routes to it. |
| `visibility` | Who may *see* the model: `GA` (everyone; the only tier exported by the public catalog endpoint), `EMPLOYEES`, `ADMINS`, or `HIDDEN`. `HIDDEN` models still **serve when explicitly routed** — that is the pre-launch testing state. Visibility never overrides `is_enabled`. |
| `min_subscription_tier` | Optional tier gate (e.g. `MAX`); enforcement arrives with the catalog-driven model picker. |
| `fallback_model_slug` | Standing replacement pointer; pre-fills the retirement flow and reserved for future automatic failover. |
| `supports_*` / `capabilities` | Capability flags (tools, JSON output, reasoning, parallel tool calls). Informational today; surfaced as warnings, never runtime-blocking. |
| `cost` | Flat `run_credits` and/or per-1M token credit rates. See the cost note below. |

> **Cost note:** block billing still reads `MODEL_COST` / `TOKEN_COST` in
> `backend/data/block_cost_config.py`. Until the catalog becomes the billing
> source, tripwire tests enforce that catalog costs and those dicts stay in
> lockstep — so a cost change means editing **both** places, and CI fails if
> they disagree. The same applies to new block-selectable models: the
> `LlmModel` enum in `backend/blocks/llm.py` remains the runtime source for
> block model selection until the catalog-driven switch lands.

`CatalogPayload.routing` holds AutoPilot's routing cells — which model serves each `(mode, tier)` combination:

```python
routing={
    "copilot": {
        "fast":     {"standard": "claude-sonnet-4-6", "advanced": "claude-opus-4-7"},
        "thinking": {"standard": "claude-sonnet-4-6", "advanced": "claude-opus-4-7"},
    },
}
```

## Updating the catalog

1. Edit `catalog.py` (add a model, change a cell, flip a flag).
2. Open a PR. Catalog-only diffs are reviewed by the `/review` bot — the integrity tests are the review.
3. Merge. CD propagates the change with the next deploy.

Two lanes:

- **Ordinary changes** (new models, metadata, visibility promotions) target `dev` and ride the normal release train.
- **Incident-speed changes** (kills, routing swaps) may use a `hotfix/*` branch targeting `master` — the base-branch check permits this — so the change deploys with CD immediately after merge. Reverting is `git revert` on the same lane.

## How AutoPilot picks a model

Each `(mode, tier)` cell resolves through three layers, top wins:

1. **LaunchDarkly `copilot-model-routing`** — per-user JSON flag returning model slugs; used for cohort experiments and rollouts. Optional: when LD is down, resolution falls through and only A/B targeting is lost.
2. **Catalog routing cell** — the PR-authored default above.
3. **`CHAT_*_MODEL` environment variables** — the bootstrap floor (see `.env.default`).

The catalog is the serve-time gate for layers 1–2: a slug that is unknown to the catalog or has `is_enabled: False` is refused — logged, reported to Sentry, and recorded for inspection — and resolution falls through to the next layer. A typo'd LD slug therefore degrades to the default instead of erroring at users. Every persisted assistant message is stamped with the model that served it and which layer picked it, which is what allows product-intelligence to compare model quality.

## Rolling out a new model

1. Add the model to the catalog with `visibility="HIDDEN"` — registered and routable, invisible in any picker or public listing.
2. Add an LD targeting rule on `copilot-model-routing` sending your test cohort (e.g. employees) to its slug.
3. Watch product-intelligence quality scores segmented by the stamped model column.
4. Graduate: flip `visibility` to `GA` and set the routing cell in a catalog PR; delete the LD rule.

## Retiring a model

Retirement has two halves:

1. **Stop it serving**: a catalog PR setting `is_enabled: False` (kill switch — beats LD routing).
2. **Migrate existing graph nodes** onto a replacement so users' agents keep working:

```bash
# dry run — prints affected node count and exits 1
python -m backend.data.llm_registry.retire <slug> --replacement <replacement-slug>

# execute (transactional, recorded, revertable)
python -m backend.data.llm_registry.retire <slug> --replacement <replacement-slug> --yes

# inspect / undo
python -m backend.data.llm_registry.retire --usage <slug>
python -m backend.data.llm_registry.retire --list
python -m backend.data.llm_registry.retire --revert <migration-id>
```

The replacement must exist in the catalog and be enabled. Every executed retirement writes a revertable `LlmModelMigration` record; only one active migration per source model is allowed at a time.

## The public catalog endpoint

`GET /api/llm/catalog` serves the catalog's public facts — GA-visibility models with their metadata and capabilities, excluding costs and routing cells (deployment-internal config). It is unauthenticated, per-IP rate-limited, and CDN-cacheable; it exists so clients (and, eventually, the catalog-driven model picker) can read the live model list without a code reference.
