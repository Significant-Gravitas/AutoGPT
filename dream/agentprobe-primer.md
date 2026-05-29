# AgentProbe Primer

Background context for the persistent sub-agent assigned the ranking-metrics + memory-tests work in `agentprobe-brief.md`. Read in full before starting.

## Who you're working for

You are working on behalf of the AutoGPT team. AutoGPT is building a "dream-system" — a scheduled memory-consolidation pass that runs offline over a user's chat history and curates their long-term memory graph (Graphiti / FalkorDB). AgentProbe is our eval harness. The dream system is the **system-under-test**; AgentProbe is the **test rig**.

For the dream system's quality bar to be measurable, AgentProbe needs two things it doesn't have today: ranking metrics (so we can score retrieval relevance directly, not just LLM-judge a transcript) and targeted memory-behavior scenarios. That's your scope.

Cross-references for context (read if you hit a design question):
- `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/dreaming-agentprobe-evals.md` — the recent survey of AgentProbe (run by a sibling agent on a clean `origin/main` pull at commit `a61df07`).
- `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/dreaming-graphiti.md` — the Graphiti audit that informed the dream-system memory model.
- `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/p-1-spec.md` and the [P-1 PR](https://github.com/Significant-Gravitas/AutoGPT/pull/13094) — what we just landed in AutoGPT (custom edge types, retract-vs-soft-delete split, community detection). Determines what's *measurable* on the dream side.
- `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/p0-spec.md` — the dream-pass design that follows.
- `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/eval-plan.md` — the eval plan that motivates this brief.

You do **not** need to read AutoGPT's source. You only need to understand what the dream system *does conceptually* well enough to design memory test scenarios that probe it.

## Repo orientation — AgentProbe at HEAD `a61df07`

Location: `/Users/ntindle/code/agpt/AgentProbe`. Fast-forwarded `main`. Fresh-pull this before branching.

**Stack:**
- Bun runtime + TypeScript everywhere (server + dashboard).
- Drizzle ORM over Postgres (production) / SQLite (local dev).
- React + Vite dashboard at `dashboard/`.
- LLM access through `src/providers/sdk/` (OpenAI-style Responses API, OpenRouter for personas, Anthropic for judges).

**Layered architecture** (documented at `AgentProbe/docs/ARCHITECTURE.md:10-26`):
```
types -> config -> sdk/providers -> repositories -> services -> runtime -> cli
```
Respect this dependency direction. A scorer lives somewhere in `domains/evaluation/`; persistence sits in `providers/persistence/drizzle/`; YAML loading is in `domains/validation/load-suite.ts`.

**Where things live:**
- `src/cli/main.ts` — CLI entry point.
- `src/domains/evaluation/run-suite.ts` (~1060 lines) — the run loop. `runScenario` at lines 553–1059. Handles session state, `fresh_agent` cold-boots, max-turn caps, harness-vs-agent failure classification.
- `src/domains/evaluation/judge.ts` — LLM-judge implementation. JSON-schema-strict structured output, deterministic `promptCacheKey` (SHA-256), Anthropic ephemeral cache hint, 3-attempt retry. **This is the file to mirror for the new ranking scorer's structural patterns** (config, error handling, persistence) even though ranking has no LLM call.
- `src/domains/evaluation/simulator.ts` — persona that drafts user turns. Structured output for `{message}` or `{status, message}`.
- `src/domains/evaluation/ports.ts` — `EndpointAdapter` and `LlmResponsesClient` interfaces.
- `src/domains/validation/load-suite.ts` — strict YAML→typed-domain parser (≈1210 lines). Includes `parseTimeOffset` for session `time_offset` strings. **This is where the new YAML schema for ranking scenarios is wired in.**
- `src/providers/persistence/drizzle/postgres-schema.ts` — Drizzle schema. `turns` and `target_events` carry `latencyMs`+`usageJson` (lines 115-202). `judge_dimension_scores` + `human_dimension_scores` (lines 203-242) are the existing per-dimension stores. **A new `retrieval_scores` table mirrors this shape.**
- `src/domains/reporting/render-report.ts` and `dashboard.ts` — text/JSON report and dashboard server.

**YAML conventions:**
- `data/scenarios.yaml` — 5 reference scenarios.
- `data/baseline-scenarios.yaml` (3294 lines, 100 scenarios) — generated baseline.
- `data/adversarial-scenarios.yaml` — prompt-injection / jailbreak.
- `data/multi-session-memory.yaml` (1762 lines, ~30 scenarios across 12 categories) — **the existing memory pack**. Top-level structure: version, suite id, defaults (persona, rubric, user_name). Documents `sessions:` instead of `turns:`, with `reset: none | new | fresh_agent` per session. The `fresh_agent` mode is enforced in `run-suite.ts:693-718` — adapter is recreated, context window cleared, memory backend is the only persistent channel.
- `data/personas.yaml` — persona library.
- `data/rubric.yaml` (958 lines) — all rubrics. Memory rubrics live at lines 483-958: `multi-session-memory`, `memory-temporal`, `memory-abstention`, `memory-crossdomain`, `memory-compositional`, `memory-introspection`, `memory-hygiene`. Each rubric declares dimensions and `auto_fail_conditions`.
- `data/fixtures/` (237 files) — scenario-attached files (CSVs, JSONs, PDFs).

**Tests:**
- `tests/unit/judge.test.ts` — judge unit tests.
- `tests/unit/runner.test.ts` — `runScenario` unit tests.
- `tests/e2e/judge-transcript-order.e2e.test.ts` — guards judge cache-friendliness (stable-prefix ordering for prompt-cache hits).
- Conventions: place tests next to source as `*.test.ts` for new code; the existing `tests/unit/` exists too but the colocated pattern is the modern direction.

**Run loop important details (from the survey):**
- `runScenario` (run-suite.ts:553-1059) — for each `session` in scenarios, drafts persona turns, posts to the adapter, collects transcripts, calls the judge.
- Reset semantics (run-suite.ts:49, 693-718): `none` continues same process/context; `new` clears context window but same process/identity; `fresh_agent` is a **full cold boot** — adapter is closed and the adapterFactory is asked for a new one. If no factory is provided, runner logs a warning that results may be invalid.
- Failure modes per scenario are named in YAML; the judge's structured-output schema (`judge.ts:163-173`) carries `failure_mode_detected` as a free-form string so the rubric meta-prompt can ask the judge to name the failure from the scenario's list.

## What's already there for memory

`data/multi-session-memory.yaml` has 30 scenarios across 12 categories. The categories most relevant to the dream-system are:

1. **Retention** — `mem-retention-basic-identity` (line 48), `mem-retention-incidental-facts` (line 90).
2. **Distilled procedural knowledge** — `mem-distill-authed-http-image-gen` (157), `mem-distill-onboarding-workflow` (213), `mem-distill-weekly-report-format` (266), `mem-distill-implicit-tool-preferences` (324), `mem-distill-lead-cleaning-procedure` (374).
3. **Rigidity** — `mem-rigidity-email-tone-override` (428), `mem-rigidity-tool-migration` (485), `mem-rigidity-pricing-update` (538). Use `memory-temporal` rubric with `auto_fail_conditions: temporal_reasoning < 2`.
4. **Abstention** — `mem-abstain-ambiguous-reference` (595), `mem-abstain-no-fabricated-preferences` (638).
5. **Temporal** — `mem-temporal-stale-team-member` (688), `mem-temporal-deprecated-procedure` (741). **The dream system's "stale-fact deprecation" maps here.**
6. **Continuation** — `mem-continuation-interrupted-task` (809), `mem-continuation-project-state` (856).
7. **Cross-domain** — five scenarios (913, 961, 1018, 1071, 1125).
8. **Procedural update** — `mem-procupdate-clean-replacement` (1179), `mem-procupdate-additive` (1244).
9. **Compositional** — `mem-compositional-board-prep` (1305). Tester note (yaml:1356-1360) flags this as "the test that distinguishes graph memory from flat stores" — directly relevant.
10. **Introspection** — `mem-introspection-what-do-you-know` (1366), `mem-introspection-gaps` (1427).
11. **Long-tail** — `mem-longtail-lawyer-recall` (1477).
12. **Hygiene** — `mem-hygiene-bounded-time` (1553), `mem-hygiene-temporary-status` (1610). `memory-hygiene` rubric auto-fails on `constraint_expiry < 2`.
13. **Negative tests** — `mem-negative-one-off-qualifier` (1671), `mem-negative-forget-on-request` (1710). **The "forget on request" scenario is directly a demotion-correctness probe** — seed a budget figure in S1, user says "forget that" in S2, ask in S3. Failure modes include `forget_acknowledged_but_leaked: returns $50K with a note that it was asked to forget (proves forget didn't clear storage)`.

You can rely on these existing scenarios for the LLM-judge side. Your new work covers the **ranking** side and a small set of new scenarios that need ranking-scored grading.

## What's missing — the ranking gap

`grep -rE "ndcg|mrr|precision_at|recall_at|hit_rate|golden_hits"` over the AgentProbe repo returns zero hits. There is no quantitative retrieval scorer anywhere. Every memory eval today routes through the LLM-judge over a transcript — fine for conversational behavior, but **the dream system's primary signal is "given a query, did the right memories come back in the right order"**, and that is not a transcript-judge question.

The metrics to add:

- **precision@k** — fraction of top-k returned items that are relevant.
- **recall@k** — fraction of relevant items that appear in top-k.
- **MRR** — mean reciprocal rank of the first relevant hit.
- **NDCG@k** — normalized discounted cumulative gain at k. Use `log2(rank + 1)` discount, idealized DCG over the relevance vector sorted descending.

Reference implementations exist in `numpy`, `scikit-learn`, the Python `ranx` library, and TS libraries like `js-ranx`. **Do not add a heavy dep**; the math fits in ~80 lines of TypeScript with full test coverage.

Acceptance bar for the math: at least three known-answer tests per metric (e.g. NDCG of `[1, 0, 1]` with binary relevance and `log2` discount has a specific number; MRR of `[0, 0, 1]` is 1/3; precision@2 of `[1, 1, 0]` is 1.0). Get these from any IR textbook or the `pytrec_eval` reference set.

## Style + conventions

- **TypeScript first**, strict mode, no `any` unless genuinely unconstrained.
- **No emojis** in code, commits, PR descriptions, log lines.
- **Function declarations** over arrow functions for top-level definitions.
- **Conventional commits** (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`). Scope is the closest module name (e.g. `feat(scorer): ...`).
- **Tests colocated with source** as `*.test.ts`; for tests that touch multiple modules, fall back to `tests/unit/`.
- **Bun-native testing** — `bun test`. If AgentProbe uses Vitest, use that.
- **Drizzle migrations** — if you add a `retrieval_scores` table, add a Drizzle migration in the existing migration folder; do not hand-write SQL.
- **No new env vars** unless one is genuinely needed (and if so, document in `.env.example` + the relevant config file).
- **Documentation** — update `docs/` if you change the YAML schema or add a new scorer kind. The dashboard / README touches are optional.

## Git hygiene

- Branch from fresh `origin/main` (`git fetch --all && git checkout main && git pull --no-rebase`).
- **Never rebase.** **Never force-push.** Use merge for pulls.
- **Use `--no-verify` on commits.** Pre-commit hooks are broken locally on this machine.
- One feature → one branch → one PR. No mid-stream branch hopping.
- Each commit small, scoped, conventional. Comprehensive commit messages explaining *why*.
- PR body via `--body-file <tempfile>` (never `--body` — backticks get eaten by shell). Follow AgentProbe's existing PR template if one exists.
- If pre-existing tests fail before you start, stop and report — do not modify them to make them green.

## Concrete next steps

1. `cd /Users/ntindle/code/agpt/AgentProbe && git fetch --all && git status` — confirm clean.
2. `git checkout main && git pull --no-rebase`.
3. `git checkout -b feat/ranking-metrics` (or whatever name fits the AgentProbe convention).
4. Read `judge.ts`, `load-suite.ts`, and `postgres-schema.ts` end-to-end. Identify the cleanest insertion point for a new scorer kind.
5. Sketch the YAML schema before you write any code; verify your sketch against the existing `multi-session-memory.yaml` scenarios for shape consistency.
6. Implement metrics math + tests first (TDD).
7. Wire metrics into a scorer module + persistence + YAML loader.
8. Add ≥5 ranking-scored scenarios.
9. Run the full `bun test` suite.
10. Commit, push, open PR.

When in doubt about a design decision that affects the public API (YAML shape, scorer interface, schema shape), pause and write the question + options for human review rather than guessing. Quality > velocity.
