---
name: block-implementer
description: "End-to-end pipeline for building a new AutoGPT Platform block from a plain-language description: intake, plan, review-plan, implement, review-impl, commit. TRIGGER when user asks to create/add/build a block, integrate a service as a block, or says 'make me a block that...'. Works from product descriptions — no technical detail required."
user-invocable: true
args: "[what the block should do] — plain language is fine, e.g. 'a block that posts a message to Slack'."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Block Implementer (Orchestrator)

This is `/feature-implementer` specialized for blocks. It runs as a **skill in the main thread** (it spawns sub-agents; agents cannot). It exists so that someone who has never read this codebase can say *"make me a block that summarizes a YouTube video"* and get a shippable, reviewed, tested block.

All step semantics — fresh agent per step, **unbounded review loops until a round returns zero findings**, executor stop-and-return handling, commit-by-pathspec — are inherited from `/feature-implementer`. Read it before running this pipeline. This file only defines what changes for blocks.

## Step 0 — Requirements intake (before any agent is spawned)

The requester may be non-technical. Translate their description into a block spec, asking **product questions only** — never pipeline or codebase questions. If something is answerable by reading code or API docs, answer it yourself instead of asking.

Establish, asking only for what's genuinely missing:

1. **Service/provider** — what external service, if any? Does a provider folder already exist under `backend/blocks/`?
2. **Action** — the single thing the block does. If the description contains "and then", plan multiple blocks and confirm the split in plain language.
3. **Auth** — does the service need an account? API key, OAuth ("log in with X"), or none? Phrase the question in terms of what the *user of the block* would have ("Do people get an API key from their settings page, or click 'Sign in with X'?").
4. **Trigger or action** — does it run when something happens (webhook trigger) or when the graph runs it?
5. **Inputs and outputs** — what does the user provide, what do they get back, in their words.

Write the spec in plain language, confirm it with the requester if anything was ambiguous, then run the pipeline. From here on the requester should not need to answer anything except true product decisions.

## Pipeline overrides

### Step 1 — Plan

Spawn the planning agent with `/feature-planner` **plus an explicit instruction to read `/add-block` first and structure the plan around its checklist**. The plan must additionally include:

- Provider folder (new or existing) and which `_config.py` auth/cost setup applies.
- The exact block class name(s), one action each.
- The test triple, including **where the real API response shape comes from** (docs URL or live call) so mocks are honest.
- OAuth/webhook wiring steps when applicable (handler + registry + `Secrets` env vars).
- A composability statement: which existing blocks plausibly feed this block's inputs and consume its outputs.

### Step 2 — Plan review

Reviewer invokes `/review-feature-plan` and must additionally check the plan against the `/add-block` registration-rules table and the composability statement. A plan whose mock shape has no cited source is a finding.

### Step 3 — Implement

Spawn `implementation-executor` with the plan **and the full text of `/add-block`** as part of its input. Block-specific verification the executor must run:

```bash
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[<NewBlock>]' -xvs
poetry run pytest backend/blocks/test/test_block.py -x
```

For OAuth providers, the executor must confirm the block actually appears in the registry with test env vars set — a missing `*_CLIENT_ID`/`*_CLIENT_SECRET` filters the block out silently.

### Step 4 — Spot-check verification

Inherited unchanged from `/feature-implementer` — the orchestrator independently re-runs the executor's verification commands rather than trusting the returned report.

### Step 5 — Implementation review

Reviewer invokes `/review-impl`; the Blocks surface lens is mandatory, with extra weight on:

- **Mock fidelity** — reviewer independently checks the mocked response shape against the provider's API docs.
- **Credentials end to end** — `_config.py` → credentials field → test credentials → (OAuth) handler registration → `Secrets`.
- **UI copy** — block description and every `SchemaField` description read like product copy for a non-technical builder user.
- **Composability** — inputs/outputs connect to plausible neighbor blocks.

### Step 6 — Commit

Conventional commit with the blocks scope: `feat(blocks): add <Provider> <action> block`. New env vars get documented in `backend/.env.default` in the same commit.

### Step 7 — Ship (when requested)

As `/feature-implementer` Step 7: `/open-pr` (target `dev`), then `/pr-polish` to drive the PR to merge-ready — both **inline in this thread**, never in a spawned agent. Block-specific additions:

- When real provider credentials are available locally, run `/pr-test` before or alongside polish: bring the stack up via docker compose and exercise the new block end-to-end (build a small graph using it, run it, check the output) — the unit harness mocks the provider, so this is the only gate that touches the real API.
- New OAuth providers warrant a `/security-review` pass before the PR (credential handling, token storage, scope breadth).
- Tell the requester the PR URL with a plain-language one-liner of what their block does — they may not read GitHub fluently.

## Final Report

As `/feature-implementer`, plus: the block ID(s), provider auth mode, env vars the deployer must set, and a one-paragraph plain-language description of what the block does — written for the person who asked for it.
