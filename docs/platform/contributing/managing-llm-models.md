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
| `slug` | Canonical model identifier (e.g. `claude-sonnet-4-6`, `gpt-5.2-2025-12-11`, `moonshotai/kimi-k2.5`). Referenced by routing cells and fallbacks. |
| `display_name` | Human-readable name shown in UIs. |
| `provider` | Who serves the model (must match a `CatalogProvider.name`). Determines which credential/API key is used. |
| `creator` | Who trained the model (display metadata; must match a `CatalogCreator.name`). |
| `context_window` / `max_output_tokens` | Token limits. |
| `price_tier` | 1 (cheapest) to 3 (most expensive); used for display. |
| `is_enabled` | **The kill switch.** A disabled model is refused at serve time — even when the routing flag routes to it. |
| `visibility` | Who may *see* the model: `GA` (everyone), `EMPLOYEES`, `ADMINS`, or `HIDDEN`. `HIDDEN` models still **serve when explicitly routed** — that is the pre-launch testing state. Informational until the catalog-driven picker lands (today a model stays out of block pickers by not having an enum line); the field is the picker's contract. Visibility never overrides `is_enabled`. |
| `fallback_model_slug` | Standing replacement pointer. A block given a switched-off model runs, and is billed for, its fallback instead, when the fallback is enabled and from the same provider; the retirement CLI also defaults `--replacement` to it. |
| `shutdown_at` | The day (UTC) the provider stops serving a switched-off model. Until then a block given it runs the model itself, at its own price; from that day on, the fallback. A node decides when it starts: one still running at midnight is billed for the model it started on, and its stats record that model (`resolved_model`). It also keeps calling that model, so a node still running after the provider stops serving it (a long agent-mode run, say) fails its later calls. Without a date the fallback applies as soon as the model is switched off. |
| `supports_*` | Capability flags (tools, JSON output, reasoning, parallel tool calls). Informational and authored opportunistically — `False` means *not asserted*, not "unsupported"; nothing consumes them at runtime yet, so only rely on authored `True` values. |
| `cost` | What users pay: flat `run_credits` and/or per-1M token **credit** rates (billing reads these). Optionally `provider_*_usd_per_1m`: what the provider charges us — the USD list price, used for in-turn cost estimates when a model is priced off its family default (e.g. Kimi K3's $3/$15). |

> **Cost note:** the catalog IS the billing source. `MODEL_METADATA`,
> `MODEL_COST`, and `TOKEN_COST` still exist as names, but they are
> **derived from the catalog at import** — there is nothing else to edit.
> One transitional artifact: `pre_catalog_costs_snapshot.json` pins the
> prices billed at the cutover, so changing a **pre-cutover** model's price
> is a deliberate two-line diff (catalog + snapshot) that shows old→new in
> review. New models never touch the snapshot, and the first legitimate
> legacy price change may simply delete the snapshot test instead
> (it is cutover proof, not a permanent fixture).

`CatalogPayload.routing` holds AutoPilot's routing cells — which model serves each `(mode, tier)` combination. **Cells ship empty**: an unset cell means the `CHAT_*_MODEL` env vars keep that combination, and *claiming* a cell is the explicit act of moving its control into the catalog:

```python
# Claiming thinking.standard — env vars keep the other three cells:
routing={
    "copilot": {
        "thinking": {"standard": "anthropic/claude-sonnet-4.6"},
    },
}
```

Cell values must be **transport-ready slugs** — the exact spelling the
serving transport accepts (OpenRouter's vendor-prefixed dot forms, as
above). The catalog's integrity tests enforce this convention.

**Cells apply only on the managed cloud deployment** (`BEHAVE_AS=cloud`).
Self-hosted installs — cloud transport or local — always resolve
the routing flag → env: a cell set for the cloud platform travels in the
shipped file but never overrides a self-hosted operator's
`CHAT_*_MODEL` configuration.

## Updating the catalog

What each change touches — this is the complete list:

| Change | You edit |
| --- | --- |
| Add a **block-selectable** model | Catalog entry + one `LLMModel` name line (`llm_registry/llm_models.py`). An import-time check refuses to boot if they drift. |
| Add a **copilot-only** model | Catalog entry. |
| Change a price (post-cutover model) | Catalog entry. |
| Change a price (pre-cutover model) | Catalog entry + its snapshot line (see cost note). |
| Kill / visibility / routing cell | Catalog entry. |

1. Edit `catalog.py` (add a model, change a cell, flip a flag).
2. Open a PR. Catalog-only diffs are reviewed by the `/review` bot — the integrity tests are the review.
3. Merge. CD propagates the change with the next deploy.

Two lanes:

- **Ordinary changes** (new models, metadata, visibility promotions) target `dev` and ride the normal release train.
- **Incident-speed changes** (kills, routing swaps) may use a `hotfix/*` branch targeting `master` — the base-branch check permits this — so the change deploys with CD immediately after merge. Reverting is `git revert` on the same lane. **Immediately merge `master` back to `dev` after a catalog hotfix**: until the back-merge lands, the next release train would silently revert your change (an emergency kill un-killing itself is the worst version of this).

Two notes. The file is public: a `HIDDEN` model is hidden from pickers, **not from anyone reading this repository** — genuinely embargoed models cannot ride this mechanism before announcement. And a catalog-only model (no enum line) simply never surfaces in blocks — it may and should still carry `cost`: copilot cost estimation uses it today and block billing picks it up automatically if the model later gains an enum line.

## How AutoPilot picks a model

Each `(mode, tier)` cell resolves through three layers, top wins:

1. **The `copilot-model-routing` feature flag** — per-user JSON flag returning model slugs; used for cohort experiments and rollouts. Optional: when the flag vendor is down, resolution falls through and only A/B targeting is lost.
2. **Catalog routing cell** — the PR-authored default above.
3. **`CHAT_*_MODEL` environment variables** — the bootstrap floor (see `.env.default`).

On the managed cloud, the catalog is the serve-time gate for layers 1–2: a slug that is unknown to the catalog or has `is_enabled: False` is refused — logged every time, reported to Sentry once per slug — and resolution falls through to the next layer. A typo'd slug in the flag therefore degrades to the default instead of erroring at users. Self-hosted installs and local transports skip the gate entirely (flag → env, their slugs are their own business). Assistant messages served by the baseline path are stamped with the model that served them and which layer picked it (`ChatMessage.model` / `routingSource`), which is what allows product-intelligence to compare model quality; the SDK path resolves through the same chain (message stamping covers the baseline path today).

## Rolling out a new model

1. Add the model to the catalog with `visibility="HIDDEN"` — registered and routable, invisible in any picker or public listing.
2. Send your test cohort (e.g. employees) to its slug on the `copilot-model-routing` PostHog flag: a variant whose JSON payload carries the slug, served by a release condition matching the cohort, in each environment's own project.
3. Watch product-intelligence quality scores segmented by the stamped model column.
4. Graduate: flip `visibility` to `GA` and set the routing cell in a catalog PR; remove the cohort's condition from the flag.

## Retiring a model

Retirement has two halves:

1. **Stop it serving**: a catalog PR setting `is_enabled: False`. On the managed cloud this is the kill switch: it beats the routing flag and hides the model from block pickers. Two caveats: if the model is also a `CHAT_*_MODEL` env default, the env floor still serves it (loudly — log + Sentry) until you change that default; and existing agent graphs referencing it keep running: on its `fallback_model_slug` when that is an enabled model of the same provider (billed as the fallback, with an info log line per run), from its `shutdown_at` when it has one; otherwise, and until then, on the model itself. So switching off a model the provider still serves, with its shutdown date set, hides it from new selection at deploy, and running workflows move on the date with no deploy. This covers values no stored-data pass reaches: a workflow input wired into a block's `model`, a preset, a schedule or an API call. Self-hosted installs and local transports skip the catalog gate, so there it doesn't stop AutoPilot serving the model: to stop that, remove its route from the `copilot-model-routing` flag and change any `CHAT_*_MODEL` default that still selects it. AutoPilot's agent generator stops building on the model too. The LLM blocks' schemas it reads list only the models the picker offers, though their enum still lists the switched-off model so stored workflows keep validating; a block with its own model list, such as Code Generation, shows that list as the block defines it. A new node it writes on a switched-off model is saved on the fallback, or on the block's default when the block doesn't offer the fallback. A node an edit leaves on the model it was saved on keeps it, even after the library has given the edited version new node ids.
2. **Migrate existing graph nodes** onto a replacement, so stored workflows name the model they run (the builder shows a switched-off model as unavailable) and a model without a fallback stops running. For a model with a `shutdown_at`, this is cleanup, any time after that date:

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

The CLI rewrites only nodes of blocks whose `model` input is an `LLMModel`. Other blocks name an input `model` too, with lists of their own (Stagehand offers a few Claude and GPT models, Codex its own GPT models), and a catalog replacement can be a value their schema refuses, so the dry run counts their nodes and the command leaves them alone. Moving those is the block's own change: Stagehand, for example, runs a node saved on Claude Sonnet 4.5 on Sonnet 4.6.

Once the provider has shut a model down, there is nothing left to revert to. Then add it to `LEGACY_MODEL_MAPPINGS` (`llm_registry/llm_models.py`) as well: `migrate_llm_models` rewrites every LLM-block node, and every preset override of one, that names it, in each install, self-hosted ones included, the next time the app starts. It logs, at INFO, how many nodes and preset overrides it moved from each stored value to its replacement, and keeps no record to revert, so map a model only once nothing can run on it. Use its catalog fallback as the target: a test checks the two agree. A replacement must not itself be a key, because each boot moves a value one step: when you retire a model that is another key's replacement, repoint that key to the new replacement in the same change (a test refuses the map otherwise).

**Small calls on reasoning models.** Reasoning and thinking tokens count against a call's output cap, so a short-answer call can spend its whole cap before it answers.
- On the native OpenAI and Anthropic paths, a call capped at 512 tokens or less (`NO_REASONING_MAX_TOKENS`) asks a model listed in `_OPENAI_LEAST_REASONING_EFFORT` or `_ANTHROPIC_NO_UPFRONT_THINKING` (`util/llm/providers.py`) for its least reasoning.
- A model that still reasons at its least effort also gets its cap raised to `SMALL_CALL_REASONING_FLOOR` (2,048). Today that is `gpt-6.1-sol` at `low`. A block's `max_tokens` of 512 or less on it can therefore return a longer answer than the cap asked for.
- When a retirement moves workflows onto a reasoning model, add the model to one of those maps, with the least value its model page lists. A value the model doesn't accept returns a 400.

## Reading the catalog from clients

There is deliberately no public catalog API: the catalog ships inside the repo, so every deployment and self-hosted install already has the exact model list its code supports. When a frontend surface needs the live list (e.g. a catalog-driven model picker), add a small authenticated route that reads the in-process registry (`backend.data.llm_registry.registry`) — don't reach for an unauthenticated endpoint; the last one existed only to bootstrap DB-seeded installs, a problem the in-repo catalog no longer has.
